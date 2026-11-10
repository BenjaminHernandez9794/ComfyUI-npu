import gc
import threading
import weakref

import torch
import safetensors.torch

import comfy.checkpoint_source
import comfy.model_management
import comfy.model_patcher
import comfy.multigpu
import comfy.weight_adapter
import comfy.utils
from comfy.patcher_extension import PatcherInjection


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.dynamic_pins = {}
        self.dynamic_vbars = {}


def _source_factory(source, disk_factory, load_device, disable_dynamic=False):
    model = TinyModel()
    model.load_state_dict(source.state_dict(), assign=True)
    patcher = comfy.model_patcher.ModelPatcher(
        model,
        load_device=load_device,
        offload_device=torch.device("cpu"),
    )
    patcher.cached_patcher_init = (disk_factory, ())
    patcher.cached_patcher_init_from_source = (
        _source_factory,
        (source, disk_factory, load_device),
    )
    return patcher


def test_checkpoint_source_returns_independent_mappings_with_shared_storage():
    weight = torch.arange(4, dtype=torch.float32).view(2, 2)
    weight.untyped_storage()._comfy_tensor_file_slice = "file-slice"
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": weight},
        {"format": "test"},
        origin="tiny.safetensors",
    )

    first = source.state_dict()
    second = source.state_dict()
    first.pop("linear.weight")

    assert "linear.weight" in second
    assert second["linear.weight"] is not weight
    assert second["linear.weight"].untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    assert second["linear.weight"].untyped_storage()._comfy_tensor_file_slice == "file-slice"
    assert source.metadata() == {"format": "test"}


def test_checkpoint_load_inside_inference_mode_returns_normal_tensors(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    safetensors.torch.save_file({"weight": torch.ones(2, 2)}, checkpoint)

    with torch.inference_mode():
        state_dict = comfy.utils.load_torch_file(str(checkpoint))

    assert not state_dict["weight"].is_inference()


def test_set_attr_param_normalizes_inference_tensor_inside_inference_mode():
    model = TinyModel()
    with torch.inference_mode():
        value = torch.full((2, 2), 3.0)
        assert value.is_inference()
        comfy.utils.set_attr_param(model, "linear.weight", value)

    assert not model.linear.weight.is_inference()
    assert torch.equal(model.linear.weight, torch.full((2, 2), 3.0))


def test_set_attr_param_restores_normal_weight_inside_inference_mode(monkeypatch):
    model = TinyModel()
    value = torch.full((2, 2), 4.0)
    parameter = torch.nn.Parameter
    constructor_modes = []

    def checked_parameter(*args, **kwargs):
        constructor_modes.append(torch.is_inference_mode_enabled())
        return parameter(*args, **kwargs)

    monkeypatch.setattr(torch.nn, "Parameter", checked_parameter)
    with torch.inference_mode():
        comfy.utils.set_attr_param(model, "linear.weight", value)

    assert constructor_modes == [False]
    assert not model.linear.weight.is_inference()
    assert torch.equal(model.linear.weight, torch.full((2, 2), 4.0))


def test_multigpu_deepclone_uses_source_and_keeps_rank_state_independent(monkeypatch):
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": torch.arange(4, dtype=torch.float32).view(2, 2)},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError("disk-backed factory must not be used for a source-backed clone")

    primary = _source_factory(source, disk_factory, torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", lambda *args, **kwargs: None)
    clone = primary.deepclone_multigpu(new_load_device=torch.device("cpu"))

    assert clone is not primary
    assert clone.model is not primary.model
    assert clone.model.linear.weight is not primary.model.linear.weight
    assert torch.equal(clone.model.linear.weight, primary.model.linear.weight)
    assert clone.backup is not primary.backup
    assert clone.patches is not primary.patches
    assert clone.model.dynamic_pins is not primary.model.dynamic_pins
    assert clone.model.dynamic_vbars is not primary.model.dynamic_vbars
    assert clone.cached_patcher_init_from_source[1][0] is source


def test_checkpoint_source_lifetime_is_bound_to_patchers():
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": torch.ones(2, 2)},
        origin="tiny.safetensors",
    )
    source_ref = weakref.ref(source)

    def disk_factory():
        raise AssertionError

    patcher = _source_factory(source, disk_factory, torch.device("cpu"))
    del source
    gc.collect()
    assert source_ref() is not None

    del patcher
    gc.collect()
    assert source_ref() is None


def test_source_backed_replica_family_opens_checkpoint_once(monkeypatch):
    open_count = 0

    def open_checkpoint():
        nonlocal open_count
        open_count += 1
        return comfy.checkpoint_source.CheckpointSource(
            {"linear.weight": torch.ones(2, 2)},
            origin="tiny.safetensors",
        )

    source = open_checkpoint()

    def disk_factory():
        return _source_factory(open_checkpoint(), disk_factory, torch.device("cpu"))

    primary = _source_factory(source, disk_factory, torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", lambda *args, **kwargs: None)
    replicas = [primary.deepclone_multigpu(new_load_device=torch.device("cpu")) for _ in range(3)]

    assert open_count == 1
    assert len({id(patcher.model) for patcher in [primary, *replicas]}) == 4
    assert all(patcher.cached_patcher_init_from_source[1][0] is source for patcher in replicas)


def test_rank_weight_update_does_not_mutate_checkpoint_source():
    source_weight = torch.ones(2, 2)
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": source_weight},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    patcher = _source_factory(source, disk_factory, torch.device("cpu"))
    replacement = torch.full((2, 2), 7.0)
    patcher.add_patches({"linear.weight": ("set", (replacement,))}, strength_patch=1.0)
    patcher.patch_weight_to_device("linear.weight", device_to=torch.device("cpu"))

    assert torch.equal(source.state_dict()["linear.weight"], source_weight)
    assert torch.equal(patcher.model.linear.weight, replacement)
    assert patcher.model.linear.weight.untyped_storage().data_ptr() != source_weight.untyped_storage().data_ptr()


def test_patch_weight_produces_normal_backup_and_parameter_inside_inference_mode():
    source_weight = torch.ones(2, 2)
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": source_weight},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    patcher = _source_factory(source, disk_factory, torch.device("cpu"))
    patcher.add_patches(
        {"linear.weight": ("set", (torch.full((2, 2), 7.0),))},
        strength_patch=1.0,
    )

    with torch.inference_mode():
        patcher.patch_weight_to_device("linear.weight", device_to=torch.device("cpu"))

    assert not patcher.backup["linear.weight"].weight.is_inference()
    assert not patcher.model.linear.weight.is_inference()


def test_patch_load_offload_cycle_stays_normal_and_preserves_source():
    source_weight = torch.arange(4, dtype=torch.float32).view(2, 2)
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": source_weight},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    patcher = _source_factory(source, disk_factory, torch.device("cpu"))
    patcher.add_patches(
        {"linear.weight": ("set", (torch.full((2, 2), 7.0),))},
        strength_patch=1.0,
    )

    with torch.inference_mode():
        for _ in range(3):
            patcher.patch_model(device_to=torch.device("cpu"))
            assert not patcher.model.linear.weight.is_inference()
            assert torch.equal(patcher.model.linear.weight, torch.full((2, 2), 7.0))
            patcher.unpatch_model(device_to=torch.device("cpu"))
            assert not patcher.model.linear.weight.is_inference()
            assert torch.equal(patcher.model.linear.weight, source_weight)

    assert torch.equal(source.state_dict()["linear.weight"], source_weight)


def test_dynamic_stage_detach_drops_device_residency_without_dropping_ram_cache(monkeypatch):
    model = TinyModel()
    device = torch.device("npu:0")
    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcher.__init__(
        patcher,
        model,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    model.dynamic_pins[device] = {}
    cache_marker = object()
    model.dynamic_pins[device]["cache_marker"] = cache_marker
    calls = []

    monkeypatch.setattr(patcher, "eject_model", lambda: None)
    monkeypatch.setattr(patcher, "model_patches_to", lambda _device: None)
    monkeypatch.setattr(
        comfy.model_patcher.ModelPatcher,
        "unpatch_model",
        lambda self, device_to=None, unpatch_weights=True: None,
    )
    monkeypatch.setattr(
        patcher,
        "partially_unload",
        lambda device_to, memory_to_free: calls.append((device_to, memory_to_free)),
    )
    monkeypatch.setattr(patcher, "get_all_callbacks", lambda _event: ())

    patcher.detach_device_residency()

    assert calls == [(None, 1e32)]
    assert model.dynamic_pins[device]["cache_marker"] is cache_marker


def test_source_backed_replicas_build_concurrently(monkeypatch):
    devices = [torch.device("npu:0"), torch.device("npu:1")]
    barrier = threading.Barrier(len(devices))

    class FakePatcher:
        def deepclone_multigpu(self, new_load_device=None, unload=True):
            assert unload is False
            barrier.wait(timeout=2)
            return new_load_device

    unload_calls = []
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", lambda patcher: unload_calls.append(patcher))
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda device: None)

    patcher = FakePatcher()
    replicas = comfy.multigpu.deepclone_patcher_to_devices(patcher, devices)

    assert replicas == devices
    assert unload_calls == [patcher]


def test_deepclone_rebinds_model_specific_injections(monkeypatch):
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": torch.ones(2, 2)},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    primary = _source_factory(source, disk_factory, torch.device("cpu"))
    rebound_models = []
    injection = PatcherInjection(
        inject=lambda _: None,
        eject=lambda _: None,
        clone_for_model=lambda model: (
            rebound_models.append(model)
            or PatcherInjection(inject=lambda _: None, eject=lambda _: None)
        ),
    )
    primary.set_injections("model_specific", [injection])
    monkeypatch.setattr(
        comfy.model_management,
        "unload_model_and_clones",
        lambda *args, **kwargs: None,
    )

    clone = primary.deepclone_multigpu(new_load_device=torch.device("cpu"))

    assert rebound_models == [clone.model]
    assert clone.injections["model_specific"][0] is not injection


def test_deepclone_bypass_injection_targets_clone_model(monkeypatch):
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": torch.ones(2, 2)},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    primary = _source_factory(source, disk_factory, torch.device("cpu"))
    adapter = comfy.weight_adapter.LoRAAdapter(
        set(),
        (torch.eye(2), torch.eye(2), None, None, None),
    )
    manager = comfy.weight_adapter.BypassInjectionManager()
    manager.add_adapter("linear.weight", adapter)
    primary.set_injections("bypass_lora", manager.create_injections(primary.model))
    monkeypatch.setattr(
        comfy.model_management,
        "unload_model_and_clones",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))

    clones = [
        primary.deepclone_multigpu(new_load_device=torch.device("cpu"))
        for _ in range(3)
    ]
    for clone in clones:
        clone.inject_model()

    assert primary.model.linear.forward.__self__ is primary.model.linear
    assert all(clone.model.linear.forward.__self__.module is clone.model.linear for clone in clones)
    assert len({id(clone.model.linear.forward.__self__) for clone in clones}) == 3


def test_vae_replicas_build_all_devices_outside_caller_inference_mode(monkeypatch):
    source = comfy.checkpoint_source.CheckpointSource(
        {"linear.weight": torch.ones(2, 2)},
        origin="tiny.safetensors",
    )

    def disk_factory():
        raise AssertionError

    primary = _source_factory(source, disk_factory, torch.device("npu:0"))

    class FakeVAE:
        patcher = primary
        first_stage_model = primary.model
        device = torch.device("npu:0")

    devices = [torch.device("npu:0"), torch.device("npu:1")]
    monkeypatch.setattr(comfy.model_management, "unload_model_and_clones", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy.model_management, "set_torch_device", lambda device: None)
    monkeypatch.setattr(comfy.model_management, "vae_offload_device", lambda: torch.device("cpu"))

    with torch.inference_mode():
        replicas = comfy.multigpu.vae_replicas(FakeVAE(), devices)

    assert [replica.device for replica in replicas] == devices
    assert all(replica is not FakeVAE for replica in replicas)
    assert all(replica.patcher is not primary for replica in replicas)
    assert all(not replica.first_stage_model.linear.weight.is_inference() for replica in replicas)
