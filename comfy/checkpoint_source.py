from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import torch


def _rank_view(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


class CheckpointSource:
    """Lifecycle-bound, read-only checkpoint tensors shared by model replicas.

    The mapping is immutable, while ``state_dict()`` returns a fresh shallow
    dictionary for loaders that rename or pop keys. Tensor storage is shared so
    opening a checkpoint once is sufficient for all replicas. The source is not
    registered globally: ModelPatcher reload factories retain it for exactly as
    long as their model family is alive.
    """

    def __init__(self, state_dict: Mapping[str, Any], metadata=None, origin=None):
        self._state_dict = MappingProxyType(dict(state_dict))
        self._metadata = MappingProxyType(dict(metadata or {}))
        self.origin = origin

    def state_dict(self) -> dict[str, Any]:
        return {key: _rank_view(value) for key, value in self._state_dict.items()}

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def __len__(self) -> int:
        return len(self._state_dict)

    def __repr__(self) -> str:
        return f"CheckpointSource(origin={self.origin!r}, tensors={len(self)})"
