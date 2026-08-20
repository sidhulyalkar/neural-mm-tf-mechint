from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ExperimentManifest:
    """Minimal provenance record for a mechanistic interpretability experiment."""

    experiment: str
    model: str
    metric: str
    seed: int
    prompt_hash: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    layers: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    python_version: str = field(default_factory=platform.python_version)
    torch_version: str = field(default_factory=lambda: torch.__version__)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    cuda_version: str | None = field(default_factory=lambda: torch.version.cuda)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination


def hash_prompts(prompts: Iterable[str]) -> str:
    """Stable SHA-256 over an ordered prompt collection."""

    payload = "\n\x1e\n".join(prompts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
