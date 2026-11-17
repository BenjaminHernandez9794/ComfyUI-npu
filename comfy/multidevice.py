from __future__ import annotations

import ctypes
import logging
import os
import threading

import torch

import comfy.multinpu


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def _current_stream(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    if device.type == "xpu":
        return torch.xpu.current_stream(device)
    if device.type == "npu":
        return torch.npu.current_stream(device)
    return None


class SequenceParallelCoordinator:
    """Coordinate variable-length sequence exchange across local devices."""

    def __init__(self, devices: list[torch.device], timeout: float = 600.0, backend: str | None = None):
        if len(devices) < 2:
            raise ValueError("Sequence parallelism requires at least two devices")
        self.devices = tuple(devices)
        self.world_size = len(devices)
        self._barrier = threading.Barrier(self.world_size, timeout=timeout)
        self._lock = threading.Lock()
        self._steps = [0] * self.world_size
        self._slots: dict[int, dict[int, tuple[torch.Tensor, ...]]] = {}
        requested_backend = (backend or os.getenv("COMFYUI_MULTI_DEVICE_BACKEND", "auto")).lower()
        if requested_backend not in ("auto", "hccl", "peer"):
            raise ValueError("Multi-device backend must be one of: auto, hccl, peer")
        self._collective = None
        if requested_backend != "peer" and all(device.type == "npu" for device in self.devices):
            try:
                self._collective = comfy.multinpu.get_hccl_communicator(self.devices)
            except (OSError, RuntimeError):
                if requested_backend == "hccl":
                    raise
                logging.exception("HCCL setup failed; falling back to peer copies")
        self.backend = "hccl_allgatherv" if self._collective is not None else "peer_copy_cat"

    def abort(self):
        self._barrier.abort()

    def exchange(self, rank: int, tensors: tuple[torch.Tensor, ...], cat_dims: tuple[int, ...]):
        if len(tensors) != len(cat_dims):
            raise ValueError("Sequence parallel exchange needs one concat dimension per tensor")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"Invalid sequence parallel rank {rank}/{self.world_size}")

        step = self._steps[rank]
        self._steps[rank] += 1
        try:
            local_device = tensors[0].device
            with self._lock:
                slot = self._slots.setdefault(step, {})
                if rank in slot:
                    raise RuntimeError(f"Sequence parallel rank {rank} posted exchange {step} twice")
                slot[rank] = tensors

            self._barrier.wait()
            with self._lock:
                posted = self._slots[step].copy()
            if len(posted) != self.world_size:
                raise RuntimeError(f"Sequence parallel exchange {step} received {len(posted)}/{self.world_size} ranks")

            if self._collective is None:
                _synchronize(local_device)
                gathered = tuple(
                    torch.cat([posted[r][i].to(local_device) for r in range(self.world_size)], dim=cat_dim)
                    for i, cat_dim in enumerate(cat_dims)
                )
            else:
                gathered_tensors = []
                for tensor_index, cat_dim in enumerate(cat_dims):
                    source = tensors[tensor_index].contiguous()
                    shapes = [tuple(posted[r][tensor_index].shape) for r in range(self.world_size)]
                    dtypes = {posted[r][tensor_index].dtype for r in range(self.world_size)}
                    if dtypes != {source.dtype}:
                        raise RuntimeError(f"Sequence parallel exchange {step} has mismatched dtypes: {dtypes}")
                    counts_list = [posted[r][tensor_index].numel() for r in range(self.world_size)]
                    offsets = [0]
                    for count in counts_list[:-1]:
                        offsets.append(offsets[-1] + count)
                    counts = (ctypes.c_uint64 * self.world_size)(*counts_list)
                    displacements = (ctypes.c_uint64 * self.world_size)(*offsets)
                    flat = torch.empty(sum(counts_list), dtype=source.dtype, device=local_device)
                    self._collective.all_gather_v(rank, source, flat, counts, displacements)
                    chunks = [flat.narrow(0, offsets[r], counts_list[r]).view(shapes[r]) for r in range(self.world_size)]
                    gathered_tensors.append(torch.cat(chunks, dim=cat_dim))
                gathered = tuple(gathered_tensors)
            stream = _current_stream(local_device)
            if stream is not None:
                stream.synchronize()
            self._barrier.wait()
            if rank == 0:
                with self._lock:
                    self._slots.pop(step, None)
            return gathered
        except Exception:
            self.abort()
            raise


class TensorParallelCoordinator:
    """Coordinate reductions for tensor-parallel shards on local devices."""

    def __init__(self, devices: list[torch.device], timeout: float = 600.0):
        if len(devices) < 2:
            raise ValueError("Tensor parallelism requires at least two devices")
        self.devices = tuple(devices)
        self.world_size = len(devices)
        self._barrier = threading.Barrier(self.world_size, timeout=timeout)
        self._lock = threading.Lock()
        self._steps = [0] * self.world_size
        self._slots: dict[int, dict[int, tuple[str, torch.Tensor]]] = {}

    def abort(self):
        self._barrier.abort()

    def _all_reduce(self, rank: int, tensor: torch.Tensor, reduction: str):
        if not 0 <= rank < self.world_size:
            raise ValueError(f"Invalid tensor parallel rank {rank}/{self.world_size}")
        if reduction not in ("sum", "max"):
            raise ValueError(f"Unsupported tensor parallel reduction: {reduction}")
        step = self._steps[rank]
        self._steps[rank] += 1
        try:
            local_device = tensor.device
            _synchronize(local_device)
            with self._lock:
                self._slots.setdefault(step, {})[rank] = (reduction, tensor)
            self._barrier.wait()
            with self._lock:
                posted_ops = self._slots[step].copy()
            if len(posted_ops) != self.world_size:
                raise RuntimeError(f"Tensor parallel reduction {step} received {len(posted_ops)}/{self.world_size} ranks")
            if {item[0] for item in posted_ops.values()} != {reduction}:
                raise RuntimeError(f"Tensor parallel reduction {step} has mismatched operations")
            posted = {posted_rank: item[1] for posted_rank, item in posted_ops.items()}
            reduced = posted[0].to(local_device).clone()
            for source_rank in range(1, self.world_size):
                source = posted[source_rank].to(local_device)
                if reduction == "sum":
                    reduced.add_(source)
                else:
                    torch.maximum(reduced, source, out=reduced)
            _synchronize(local_device)
            self._barrier.wait()
            if rank == 0:
                with self._lock:
                    self._slots.pop(step, None)
            return reduced
        except Exception:
            self.abort()
            raise

    def all_reduce_sum(self, rank: int, tensor: torch.Tensor):
        return self._all_reduce(rank, tensor, "sum")

    def all_reduce_max(self, rank: int, tensor: torch.Tensor):
        return self._all_reduce(rank, tensor, "max")
