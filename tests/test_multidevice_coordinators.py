import threading
import time

import torch

import comfy.multidevice
import comfy.multigpu
import comfy.multinpu
import comfy.ops
from comfy.ldm.lightricks.model import CrossAttention
from comfy.ldm.lightricks.av_model import CompressedTimestep


def _run_ranks(world_size, fn):
    outputs = [None] * world_size
    errors = [None] * world_size

    def run(rank):
        try:
            outputs[rank] = fn(rank)
        except Exception as error:
            errors[rank] = error

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == [None] * world_size
    return outputs


def test_sequence_parallel_peer_exchange_variable_lengths():
    devices = [torch.device("cpu"), torch.device("cpu")]
    coordinator = comfy.multidevice.SequenceParallelCoordinator(devices, backend="peer")

    outputs = _run_ranks(
        2,
        lambda rank: coordinator.exchange(
            rank,
            (torch.full((rank + 1, 2), rank + 1.0),),
            (0,),
        )[0],
    )

    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    assert all(torch.equal(output, expected) for output in outputs)


def test_tensor_parallel_sum_and_max():
    devices = [torch.device("cpu"), torch.device("cpu")]
    coordinator = comfy.multidevice.TensorParallelCoordinator(devices)

    sums = _run_ranks(2, lambda rank: coordinator.all_reduce_sum(rank, torch.tensor([rank + 1.0])))
    maximums = _run_ranks(2, lambda rank: coordinator.all_reduce_max(rank, torch.tensor([rank + 1.0])))

    assert all(torch.equal(output, torch.tensor([3.0])) for output in sums)
    assert all(torch.equal(output, torch.tensor([2.0])) for output in maximums)


def test_hccl_communicator_keeps_rank_devices(monkeypatch):
    class FakeFunction:
        def __call__(self, *args):
            return 0

    class FakeHccl:
        HcclCommInitAll = FakeFunction()
        HcclAllGatherV = FakeFunction()

    monkeypatch.setattr(comfy.multinpu.ctypes, "CDLL", lambda _: FakeHccl())
    devices = (torch.device("npu:0"), torch.device("npu:1"))

    communicator = comfy.multinpu.HcclLocalCommunicator(devices)

    assert communicator.devices == devices
    assert communicator.device_indices == (0, 1)
    assert communicator.world_size == 2


def test_multigpu_thread_pool_disables_grad_without_creating_inference_tensors():
    device = torch.device("cpu")
    pool = comfy.multigpu.MultiGPUThreadPool([device])
    try:
        with torch.inference_mode():
            inference_tensor = torch.ones(1)
            pool.submit(
                device,
                lambda tensor: (
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                    (tensor * 2).is_inference(),
                ),
                inference_tensor,
            )

        pool.submit(
            device,
            lambda: (torch.is_inference_mode_enabled(), torch.is_grad_enabled()),
        )

        inference_result, inference_error = pool.get_result(device)
        grad_result, grad_error = pool.get_result(device)
    finally:
        pool.shutdown()

    assert inference_error is None
    assert inference_result == (False, False, False)
    assert grad_error is None
    assert grad_result == (False, True)


def test_multigpu_thread_pool_runs_devices_concurrently():
    devices = [torch.device("cpu:0"), torch.device("cpu:1")]
    pool = comfy.multigpu.MultiGPUThreadPool(devices)
    barrier = threading.Barrier(len(devices))
    try:
        for device in devices:
            pool.submit(device, barrier.wait, 2)
        results = [pool.get_result(device) for device in devices]
    finally:
        pool.shutdown()

    assert all(error is None for _, error in results)


def test_multigpu_thread_pool_keeps_device_queue_order():
    device = torch.device("cpu")
    pool = comfy.multigpu.MultiGPUThreadPool([device])
    order = []
    try:
        pool.submit(device, lambda: (time.sleep(0.01), order.append(1)))
        pool.submit(device, lambda: order.append(2))
        first = pool.get_result(device)
        second = pool.get_result(device)
    finally:
        pool.shutdown()

    assert first[1] is None
    assert second[1] is None
    assert order == [1, 2]


def test_ltx_sequence_parallel_attention_matches_full_sequence():
    torch.manual_seed(7)
    attention = CrossAttention(
        query_dim=16,
        heads=2,
        dim_head=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init,
    )
    # disable_weight_init leaves parameters uninitialized; give them real values
    with torch.no_grad():
        for param in attention.parameters():
            param.normal_(0.0, 0.2)
    x = torch.randn(1, 7, 16)
    expected = attention(x, transformer_options={})

    devices = [torch.device("cpu"), torch.device("cpu")]
    coordinator = comfy.multidevice.SequenceParallelCoordinator(devices, backend="peer")

    def run_rank(rank):
        start = x.shape[1] * rank // 2
        stop = x.shape[1] * (rank + 1) // 2
        return attention(
            x[:, start:stop].contiguous(),
            transformer_options={
                "sequence_parallel_context": {
                    "coordinator": coordinator,
                    "rank": rank,
                    "world_size": 2,
                }
            },
            sequence_parallel=True,
        )

    actual = torch.cat(_run_ranks(2, run_rank), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_ltx_sequence_parallel_keeps_scalar_timestep_broadcastable():
    timestep = CompressedTimestep(torch.randn(1, 1, 12), patches_per_frame=209)

    sliced = timestep.slice_sequence(418, 627)

    assert sliced.expand().shape == (1, 1, 12)
