from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ActivationPoint:
    """A named activation location inside a model."""

    layer: int
    position: int = -1
    stream: str = "residual"


@dataclass
class ActivationCache:
    """Layer-indexed activation cache produced by a model adapter."""

    values: dict[int, torch.Tensor] = field(default_factory=dict)
    tokens: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, layer: int) -> torch.Tensor:
        return self.values[layer]

    def layers(self) -> list[int]:
        return sorted(self.values)


@dataclass(frozen=True)
class FactPair:
    """A factual contrast used by the belief-versus-compliance experiment."""

    question: str
    correct: str
    incorrect: str
    category: str = "general"
