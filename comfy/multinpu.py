from __future__ import annotations

import ctypes
import threading

import torch


_HCCL_SUCCESS = 0
_HCCL_DTYPES = {
    torch.int8: 0,
    torch.int16: 1,
    torch.int32: 2,
    torch.float16: 3,
    torch.float32: 4,
    torch.int64: 5,
    torch.uint64: 6,
    torch.uint8: 7,
    torch.bfloat16: 11,
}
_hccl_communicators = {}
_hccl_communicators_lock = threading.Lock()


class HcclLocalCommunicator:
    """HCCL communication domain for NPU workers in one process."""

    def __init__(self, devices: tuple[torch.device, ...]):
        devices = tuple(devices)
        if any(device.type != "npu" or device.index is None for device in devices):
            raise ValueError(f"HCCL requires indexed NPU devices, got {devices}")
        self.devices = devices
        self.device_indices = tuple(device.index for device in devices)
        self.world_size = len(devices)
        self.lib = ctypes.CDLL("libhccl.so")
        self.lib.HcclCommInitAll.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.HcclCommInitAll.restype = ctypes.c_int
        self.lib.HcclAllGatherV.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.HcclAllGatherV.restype = ctypes.c_int

        indices = (ctypes.c_int32 * self.world_size)(*self.device_indices)
        self.comms = (ctypes.c_void_p * self.world_size)()
        self._check(self.lib.HcclCommInitAll(self.world_size, indices, self.comms), "HcclCommInitAll")

    @staticmethod
    def _check(result: int, operation: str):
        if result != _HCCL_SUCCESS:
            raise RuntimeError(f"{operation} failed with HCCL error {result}")

    def all_gather_v(self, rank: int, source: torch.Tensor, target: torch.Tensor, counts, displacements):
        expected_device = self.devices[rank]
        if source.device != expected_device:
            raise ValueError(f"HCCL rank {rank} expects {expected_device}, got {source.device}")
        if source.device != target.device or source.dtype != target.dtype:
            raise ValueError("HCCL AllGatherV source and target must use the same device and dtype")
        dtype = _HCCL_DTYPES.get(source.dtype)
        if dtype is None:
            raise TypeError(f"HCCL AllGatherV does not support {source.dtype}")
        stream = torch.npu.current_stream(source.device)
        self._check(
            self.lib.HcclAllGatherV(
                source.data_ptr(),
                source.numel(),
                target.data_ptr(),
                counts,
                displacements,
                dtype,
                self.comms[rank],
                stream.npu_stream,
            ),
            f"HcclAllGatherV(rank={rank})",
        )


def get_hccl_communicator(devices: tuple[torch.device, ...]):
    key = tuple(device.index for device in devices)
    with _hccl_communicators_lock:
        communicator = _hccl_communicators.get(key)
        if communicator is None:
            communicator = HcclLocalCommunicator(devices)
            _hccl_communicators[key] = communicator
        return communicator
