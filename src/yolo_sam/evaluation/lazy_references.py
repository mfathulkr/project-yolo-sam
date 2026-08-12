from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np


class LazyMaskReferences:
    """Keep encoded references compact and decode one instance at a time."""

    def __init__(
        self,
        encoded: Mapping[str, Any],
        decoder: Callable[[Any], np.ndarray],
    ) -> None:
        self._encoded = dict(encoded)
        self._decoder = decoder

    def __len__(self) -> int:
        return len(self._encoded)

    def __iter__(self) -> Iterator[str]:
        return iter(self._encoded)

    def __contains__(self, instance_id: object) -> bool:
        return instance_id in self._encoded

    def mask(self, instance_id: str) -> np.ndarray:
        if instance_id not in self._encoded:
            raise KeyError(instance_id)
        mask = np.asarray(self._decoder(self._encoded[instance_id]), dtype=bool)
        if mask.ndim != 2:
            raise ValueError(
                f"Decoded reference mask must be 2-D, got shape={mask.shape}"
            )
        return mask

    @classmethod
    def empty(cls) -> "LazyMaskReferences":
        return cls({}, lambda _: np.zeros((0, 0), dtype=bool))
