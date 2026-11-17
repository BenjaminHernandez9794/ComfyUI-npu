from __future__ import annotations
import copy
import concurrent.futures
import itertools
import queue
import threading
import time
import torch
import logging

from collections import namedtuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import comfy.utils
import comfy.patcher_extension
import comfy.model_management
import comfy.model_prefetch
from comfy.multidevice import SequenceParallelCoordinator, TensorParallelCoordinator


class MultiGPUThreadPool:
    """Persistent thread pool for multi-GPU work distribution.

    Maintains one worker thread per extra GPU device. Each thread calls
    set_torch_device() once at startup so that compiled kernel caches
    (inductor/triton) stay warm across diffusion steps.
    """

    def __init__(self, devices: list[torch.device]):
        self._workers: list[threading.Thread] = []
        self._work_queues: dict[torch.device, queue.Queue] = {}
        self._result_queues: dict[torch.device, queue.Queue] = {}

        for device in devices:
            wq = queue.Queue()
            rq = queue.Queue()
            self._work_queues[device] = wq
            self._result_queues[device] = rq
            t = threading.Thread(target=self._worker_loop, args=(device, wq, rq), daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self, device: torch.device, work_q: queue.Queue, result_q: queue.Queue):
        try:
            comfy.model_management.set_torch_device(device)
        except Exception as e:
            logging.error(f"MultiGPUThreadPool: failed to set device {device}: {e}")
            while True:
                item = work_q.get()
                if item is None:
                    return
                result_q.put((None, e))
            return
        try:
            while True:
                item = work_q.get()
                if item is None:
                    break
                fn, args, kwargs, grad_enabled = item
                try:
                    with torch.set_grad_enabled(grad_enabled):
                        result = fn(*args, **kwargs)
                    result_q.put((result, None))
                except comfy.model_management.InterruptProcessingException as e:
                    result_q.put((None, e))
                except Exception as e:
                    logging.exception("Multi-device worker failed on %s", device)
                    result_q.put((None, e))
        finally:
            comfy.model_prefetch.cleanup_malloc_graph()

    def submit(self, device: torch.device, fn, *args, **kwargs):
        self._work_queues[device].put(
            (fn, args, kwargs, torch.is_grad_enabled())
        )

    def get_result(self, device: torch.device):
        return self._result_queues[device].get()

    @property
    def devices(self) -> list[torch.device]:
        return list(self._work_queues.keys())

    def shutdown(self):
        for wq in self._work_queues.values():
            wq.put(None)  # sentinel
        for t in self._workers:
            t.join(timeout=5.0)


def clone_patcher_to_device(patcher: ModelPatcher, device: torch.device, reuse_loaded=True):
    """Clone a patcher onto one device, reusing a resident clone family when possible."""
    if patcher.load_device == device:
        return patcher.clone()
    if reuse_loaded:
        for loaded in comfy.model_management.loaded_models():
            if (
                loaded is not None
                and loaded.load_device == device
                and loaded.clone_base_uuid == patcher.clone_base_uuid
            ):
                return patcher.clone(model_override=loaded.get_clone_model_override())
    return patcher.deepclone_multigpu(new_load_device=device)


def deepclone_patcher_to_devices(patcher: ModelPatcher, devices, reload_model_options=None):
    """Build independent source-backed replicas concurrently on NPU devices."""
    devices = tuple(devices)
    if not devices:
        return []
    if reload_model_options is None:
        reload_options = (None,) * len(devices)
    else:
        reload_options = tuple(reload_model_options)
        if len(reload_options) != len(devices):
            raise ValueError("reload_model_options must contain one entry per device")
    if len(devices) == 1 or any(device.type != "npu" for device in devices):
        return [
            patcher.deepclone_multigpu(new_load_device=device, **(
                {"reload_model_options": options} if options is not None else {}
            ))
            for device, options in zip(devices, reload_options)
        ]

    comfy.model_management.unload_model_and_clones(patcher)
    pool = MultiGPUThreadPool(list(devices))
    try:
        for device, options in zip(devices, reload_options):
            kwargs = {"new_load_device": device, "unload": False}
            if options is not None:
                kwargs["reload_model_options"] = options
            pool.submit(device, patcher.deepclone_multigpu, **kwargs)
        results = []
        for device in devices:
            replica, error = pool.get_result(device)
            results.append((replica, error))
        for _, error in results:
            if error is not None:
                raise error
        return [replica for replica, _ in results]
    finally:
        pool.shutdown()


def component_patchers(*components):
    """Yield patchers owned by pipeline components, including nested models."""
    for component in components:
        patcher = getattr(component, "patcher", component)
        if patcher is None:
            continue
        yield patcher
        yield from patcher.get_nested_additional_models()


def prepare_stage_devices(*components, stage_name="pipeline stage"):
    """Make room for one pipeline stage while retaining its model families."""
    stage_start = time.perf_counter()
    device_patchers = {}
    for patcher in component_patchers(*components):
        device_patchers.setdefault(patcher.load_device, []).append(patcher)
    npu_devices = [device for device in device_patchers if device.type == "npu"]
    if len(npu_devices) > 1:
        claimed = comfy.model_management.claim_device_models_except(device_patchers)

        def unload_device(device):
            device_start = time.perf_counter()
            loaded_models = claimed[device]
            with comfy.model_management.cuda_device_context(device):
                comfy.model_management.synchronize()
                for loaded_model in loaded_models:
                    loaded_model.model_stage_unload()
                comfy.model_management.soft_empty_cache()
            elapsed = time.perf_counter() - device_start
            logging.info(
                "Pipeline stage prepare: stage=%s device=%s unloaded=%d elapsed=%.2fs",
                stage_name,
                device,
                len(loaded_models),
                elapsed,
            )
            return elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_patchers)) as executor:
            futures = {
                device: executor.submit(unload_device, device)
                for device in device_patchers
            }
            for device, future in futures.items():
                try:
                    future.result()
                except Exception:
                    logging.exception(
                        "Pipeline stage prepare failed: stage=%s device=%s",
                        stage_name,
                        device,
                    )
                    raise
    else:
        for device, patchers in device_patchers.items():
            device_start = time.perf_counter()
            with comfy.model_management.cuda_device_context(device):
                comfy.model_management.synchronize()
                unloaded = comfy.model_management.unload_device_models_except(device, patchers)
                comfy.model_management.soft_empty_cache()
            logging.info(
                "Pipeline stage prepare: stage=%s device=%s unloaded=%d elapsed=%.2fs",
                stage_name,
                device,
                len(unloaded),
                time.perf_counter() - device_start,
            )
    elapsed = time.perf_counter() - stage_start
    logging.info(
        "Pipeline stage prepare completed: stage=%s devices=%d elapsed=%.2fs",
        stage_name,
        len(device_patchers),
        elapsed,
    )
    return elapsed


def offload_components(*components, all_devices=True, empty_cache=False, stage_name="pipeline stage"):
    """Unload complete clone families without destroying their CPU reload sources."""
    stage_start = time.perf_counter()
    families = set()
    devices = set()
    for patcher in component_patchers(*components):
        devices.add(patcher.load_device)
        if patcher.clone_base_uuid in families:
            continue
        families.add(patcher.clone_base_uuid)
        with comfy.model_management.cuda_device_context(patcher.load_device):
            comfy.model_management.synchronize()
            comfy.model_management.unload_model_and_clones(patcher, all_devices=all_devices)
    if empty_cache:
        empty_device_caches(devices or comfy.model_management.get_all_torch_devices(exclude_current=False))
    elapsed = time.perf_counter() - stage_start
    logging.info(
        "Pipeline stage offload completed: stage=%s families=%d devices=%d elapsed=%.2fs",
        stage_name,
        len(families),
        len(devices),
        elapsed,
    )
    return elapsed


def empty_device_caches(devices):
    """Release allocator reservations on every selected accelerator."""
    for device in dict.fromkeys(devices):
        if device.type == "cpu":
            continue
        with comfy.model_management.cuda_device_context(device):
            comfy.model_management.synchronize()
            comfy.model_management.soft_empty_cache()


def place_clip(clip, device: torch.device, reuse_loaded=True):
    placed = clip.clone()
    placed.patcher = clone_patcher_to_device(clip.patcher, device, reuse_loaded=reuse_loaded)
    placed.patcher.load_device = device
    placed.cond_stage_model = placed.patcher.model
    return placed


def place_vae(vae, device: torch.device, reuse_loaded=True):
    placed = copy.copy(vae)
    placed.patcher = clone_patcher_to_device(vae.patcher, device, reuse_loaded=reuse_loaded)
    placed.patcher.load_device = device
    placed.patcher.offload_device = comfy.model_management.vae_offload_device()
    placed.first_stage_model = placed.patcher.model
    placed.device = device
    return placed


def vae_replicas(vae, devices):
    """Create one independently loadable VAE wrapper per device."""
    cached = getattr(vae, "_multidevice_replicas", {})
    replicas_by_device = {}
    missing_devices = []
    for device in devices:
        replica = cached.get(device)
        if (
            replica is None
            or replica.patcher.clone_base_uuid != vae.patcher.clone_base_uuid
        ):
            missing_devices.append(device)
        else:
            replicas_by_device[device] = replica

    for device, patcher in zip(
        missing_devices,
        deepclone_patcher_to_devices(vae.patcher, missing_devices),
    ):
        replica = copy.copy(vae)
        replica.patcher = patcher
        replica.patcher.load_device = device
        replica.patcher.offload_device = comfy.model_management.vae_offload_device()
        replica.first_stage_model = patcher.model
        replica.device = device
        cached[device] = replica
        replicas_by_device[device] = replica

    vae._multidevice_replicas = cached
    return [replicas_by_device[device] for device in devices]


def _scale_value(scale, value):
    return scale(value) if callable(scale) else scale * value


def _tiled_positions(samples, tile, overlap):
    positions = [
        range(0, samples.shape[dim + 2] - overlap[dim], tile[dim] - overlap[dim])
        if samples.shape[dim + 2] > tile[dim] else [0]
        for dim in range(len(tile))
    ]
    for batch in range(samples.shape[0]):
        for candidate in itertools.product(*positions):
            starts = tuple(
                max(0, min(samples.shape[dim + 2] - overlap[dim], candidate[dim]))
                for dim in range(len(tile))
            )
            lengths = tuple(
                min(tile[dim], samples.shape[dim + 2] - starts[dim])
                for dim in range(len(tile))
            )
            yield batch, starts, lengths


def _decode_vae_tile(vae, tile):
    with comfy.model_management.cuda_device_context(vae.device):
        decoded = vae.first_stage_model.decode(tile.to(device=vae.device, dtype=vae.vae_dtype))
        decoded = decoded.to(dtype=vae.vae_output_dtype(), device=vae.output_device)
        comfy.model_management.synchronize()
        return decoded


@torch.inference_mode()
def decode_tiled_multidevice(vae, samples, devices, tile, overlap):
    """Decode independent VAE tiles on multiple devices and feather-merge them."""
    devices = tuple(devices)
    if not devices:
        raise ValueError("Multi-device VAE decode requires at least one device")
    if len(tile) != samples.ndim - 2 or len(overlap) != len(tile):
        raise ValueError("VAE tile geometry does not match the latent dimensions")
    if any(tile_dim <= overlap_dim for tile_dim, overlap_dim in zip(tile, overlap)):
        raise ValueError("Each VAE tile dimension must be larger than its overlap")

    replicas = vae_replicas(vae, devices)
    bounded_shape = vae._tile_bounded_shape(
        samples.shape,
        tile[-1] if len(tile) >= 1 else None,
        tile[-2] if len(tile) >= 2 else None,
        tile[-3] if len(tile) >= 3 else None,
    )
    memory_required = vae.memory_used_decode(bounded_shape, vae.vae_dtype)
    comfy.model_management.load_models_gpu(
        [replica.patcher for replica in replicas],
        memory_required=memory_required,
        force_full_load=vae.disable_offload,
    )

    assignments = {device: [] for device in devices}
    tasks = list(_tiled_positions(samples, tile, overlap))
    for index, task in enumerate(tasks):
        assignments[devices[index % len(devices)]].append((index, task))

    pool = MultiGPUThreadPool(list(devices))
    try:
        for replica in replicas:
            device_tasks = assignments[replica.device]

            def decode_assigned(replica=replica, device_tasks=device_tasks):
                results = []
                for index, (batch, starts, lengths) in device_tasks:
                    latent_tile = samples[batch:batch + 1]
                    for dim, (start, length) in enumerate(zip(starts, lengths)):
                        latent_tile = latent_tile.narrow(dim + 2, start, length)
                    results.append((index, batch, starts, _decode_vae_tile(replica, latent_tile)))
                return results

            pool.submit(replica.device, decode_assigned)

        decoded_tiles = []
        for replica in replicas:
            result, error = pool.get_result(replica.device)
            if error is not None:
                raise error
            decoded_tiles.extend(result)
    finally:
        pool.shutdown()

    scales = vae.upscale_ratio
    if not isinstance(scales, (tuple, list)):
        scales = (scales,) * len(tile)
    else:
        scales = tuple(scales[-len(tile):])
    index_scales = vae.upscale_index_formula
    if not isinstance(index_scales, (tuple, list)):
        index_scales = (index_scales,) * len(tile)
    else:
        index_scales = tuple(index_scales[-len(tile):])
    output_shape = [samples.shape[0], vae.output_channels]
    output_shape.extend(round(_scale_value(scale, size)) for scale, size in zip(scales, samples.shape[2:]))
    output = torch.zeros(output_shape, device=vae.output_device)
    divisor = torch.zeros([samples.shape[0], 1, *output_shape[2:]], device=vae.output_device)

    for _, batch, starts, decoded in sorted(decoded_tiles):
        output_starts = [round(_scale_value(scale, start)) for scale, start in zip(index_scales, starts)]
        mask = torch.ones([1, 1, *decoded.shape[2:]], device=vae.output_device)
        for dim, overlap_dim in enumerate(overlap):
            feather = round(_scale_value(scales[dim], overlap_dim))
            if feather >= mask.shape[dim + 2]:
                continue
            for index in range(feather):
                alpha = (index + 1) / feather
                mask.narrow(dim + 2, index, 1).mul_(alpha)
                mask.narrow(dim + 2, mask.shape[dim + 2] - 1 - index, 1).mul_(alpha)

        output_view = output[batch:batch + 1]
        divisor_view = divisor[batch:batch + 1]
        decoded_view = decoded
        mask_view = mask
        for dim, start in enumerate(output_starts):
            length = min(decoded_view.shape[dim + 2], output_view.shape[dim + 2] - start)
            output_view = output_view.narrow(dim + 2, start, length)
            divisor_view = divisor_view.narrow(dim + 2, start, length)
            if length < decoded_view.shape[dim + 2]:
                decoded_view = decoded_view.narrow(dim + 2, 0, length)
                mask_view = mask_view.narrow(dim + 2, 0, length)
        output_view.add_(decoded_view * mask_view)
        divisor_view.add_(mask_view)

    return vae.process_output(output.div_(divisor)).movedim(1, -1)


class GPUOptions:
    def __init__(self, device_index: int, relative_speed: float):
        self.device_index = device_index
        self.relative_speed = relative_speed

    def clone(self):
        return GPUOptions(self.device_index, self.relative_speed)

    def create_dict(self):
        return {
            "relative_speed": self.relative_speed
        }

class GPUOptionsGroup:
    def __init__(self):
        self.options: dict[int, GPUOptions] = {}

    def add(self, info: GPUOptions):
        self.options[info.device_index] = info

    def clone(self):
        c = GPUOptionsGroup()
        for opt in self.options.values():
            c.add(opt)
        return c

    def register(self, model: ModelPatcher):
        opts_dict = {}
        # get devices that are valid for this model
        devices: list[torch.device] = [model.load_device]
        for extra_model in model.get_additional_models_with_key("multigpu"):
            extra_model: ModelPatcher
            devices.append(extra_model.load_device)
        # create dictionary with actual device mapped to its GPUOptions
        device_opts_list: list[GPUOptions] = []
        for device in devices:
            device_opts = self.options.get(device.index, GPUOptions(device_index=device.index, relative_speed=1.0))
            opts_dict[device] = device_opts.create_dict()
            device_opts_list.append(device_opts)
        # make relative_speed relative to 1.0
        min_speed = min([x.relative_speed for x in device_opts_list])
        for value in opts_dict.values():
            value['relative_speed'] /= min_speed
        model.model_options['multigpu_options'] = opts_dict


def create_multigpu_deepclones(
    model: ModelPatcher,
    max_gpus: int,
    gpu_options: GPUOptionsGroup=None,
    reuse_loaded=False,
    reload_model_options=None,
):
    'Prepare ModelPatcher to contain deepclones of its BaseModel and related properties.'
    model = model.clone()
    # check if multigpu is already prepared - get the load devices from them if possible to exclude
    skip_devices = set()
    multigpu_models = model.get_additional_models_with_key("multigpu")
    if len(multigpu_models) > 0:
        for mm in multigpu_models:
            skip_devices.add(mm.load_device)
    skip_devices = list(skip_devices)

    # Exclude the primary model's actual device, not the global current device:
    # after SelectModelDevice(gpu:N) the primary may not live on the process's
    # current CUDA device, and excluding the wrong device picks bad extras.
    all_devices = comfy.model_management.get_all_torch_devices(exclude_current=False)
    full_extra_devices = [d for d in all_devices if d != model.load_device]
    limit_extra_devices = full_extra_devices[:max_gpus-1]
    extra_devices = limit_extra_devices.copy()
    # exclude skipped devices
    for skip in skip_devices:
        if skip in extra_devices:
            extra_devices.remove(skip)
    # create new deepclones
    if len(extra_devices) > 0:
        device_patchers = {}
        for device in extra_devices:
            if reuse_loaded:
                # Only reuse a previously-loaded MultiGPU clone. A SelectModelDevice
                # patcher on the same device shares clone_base_uuid but has
                # is_multigpu_base_clone=False, which would later be filtered out by
                # prepare_model_patcher_multigpu_clones() and silently shrink the
                # work split back to one GPU.
                loaded_models: list[ModelPatcher] = comfy.model_management.loaded_models()
                for lm in loaded_models:
                    if lm.model is None:
                        continue
                    if lm.load_device != device:
                        continue
                    if lm.clone_base_uuid != model.clone_base_uuid:
                        continue
                    if not getattr(lm, "is_multigpu_base_clone", False):
                        continue
                    device_patchers[device] = lm.clone()
                    logging.info(f"Reusing loaded multigpu deepclone of {device_patchers[device].model.__class__.__name__} for {device}")
                    break

        missing_devices = [device for device in extra_devices if device not in device_patchers]
        for device, device_patcher in zip(
            missing_devices,
            deepclone_patcher_to_devices(
                model,
                missing_devices,
                reload_model_options=(
                    [reload_model_options] * len(missing_devices)
                    if reload_model_options is not None else None
                ),
            ),
        ):
            device_patchers[device] = device_patcher

        for device in extra_devices:
            device_patcher = device_patchers[device]
            # Always flag the clone; whether reused or freshly deepcloned, it must
            # advertise itself as a MultiGPU base clone so the cond scheduler picks
            # it up in prepare_model_patcher_multigpu_clones().
            device_patcher.is_multigpu_base_clone = True
            multigpu_models = model.get_additional_models_with_key("multigpu")
            multigpu_models.append(device_patcher)
            model.set_additional_models("multigpu", multigpu_models)
        model.match_multigpu_clones()
        if gpu_options is None:
            gpu_options = GPUOptionsGroup()
        gpu_options.register(model)
    else:
        logging.info("No extra torch devices need initialization, skipping initializing MultiGPU Work Units.")
    # only keep model clones that don't go 'past' the intended max_gpu count;
    # this prunes any inherited multigpu clones whose load_device is no longer allowed
    # when max_gpus is lowered between runs.
    allowed_devices = set(limit_extra_devices)
    allowed_devices.add(model.load_device)
    multigpu_models = model.get_additional_models_with_key("multigpu")
    new_multigpu_models = [m for m in multigpu_models if m.load_device in allowed_devices]
    if len(new_multigpu_models) != len(multigpu_models):
        model.set_additional_models("multigpu", new_multigpu_models)
        model.match_multigpu_clones()
    return model


LoadBalance = namedtuple('LoadBalance', ['work_per_device', 'idle_time'])
def load_balance_devices(model_options: dict[str], total_work: int, return_idle_time=False, work_normalized: int=None):
    'Optimize work assigned to different devices, accounting for their relative speeds and splittable work.'
    opts_dict = model_options['multigpu_options']
    devices = list(model_options['multigpu_clones'].keys())
    speed_per_device = []
    work_per_device = []
    # get sum of each device's relative_speed
    total_speed = 0.0
    for opts in opts_dict.values():
        total_speed += opts['relative_speed']
    # get relative work for each device;
    # obtained by w = (W*r)/R
    for device in devices:
        relative_speed = opts_dict[device]['relative_speed']
        relative_work = (total_work*relative_speed) / total_speed
        speed_per_device.append(relative_speed)
        work_per_device.append(relative_work)
    # relative work must be expressed in whole numbers, but likely is a decimal;
    # perform rounding while maintaining total sum equal to total work (sum of relative works)
    work_per_device = round_preserved(work_per_device)
    dict_work_per_device = {}
    for device, relative_work in zip(devices, work_per_device):
        dict_work_per_device[device] = relative_work
    if not return_idle_time:
        return LoadBalance(dict_work_per_device, None)
    # divide relative work by relative speed to get estimated completion time of said work by each device;
    # time here is relative and does not correspond to real-world units
    completion_time = [w/r for w,r in zip(work_per_device, speed_per_device)]
    # calculate relative time spent by the devices waiting on each other after their work is completed
    idle_time = abs(min(completion_time) - max(completion_time))
    # if need to compare work idle time, need to normalize to a common total work
    if work_normalized:
        idle_time *= (work_normalized/total_work)

    return LoadBalance(dict_work_per_device, idle_time)

def round_preserved(values: list[float]):
    'Round all values in a list, preserving the combined sum of values.'
    # get floor of values; casting to int does it too
    floored = [int(x) for x in values]
    total_floored = sum(floored)
    # get remainder to distribute
    remainder = round(sum(values)) - total_floored
    # pair values with fractional portions
    fractional = [(i, x-floored[i]) for i, x in enumerate(values)]
    # sort by fractional part in descending order
    fractional.sort(key=lambda x: x[1], reverse=True)
    # distribute the remainder
    for i in range(remainder):
        index = fractional[i][0]
        floored[index] += 1
    return floored
