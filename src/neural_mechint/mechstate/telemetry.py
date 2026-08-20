from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .state import MechanisticTrajectory


@dataclass
class StateReceipt:
    """Serializable audit receipt for one guarded inference."""
    decision: str
    reasons: list[str]
    trajectory: MechanisticTrajectory
    intervention: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def receipt_id(self) -> str:
        payload = json.dumps(self.to_dict(include_id=False), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        data = {"decision": self.decision, "reasons": self.reasons, "trajectory": self.trajectory.to_dict(), "intervention": self.intervention, "metrics": self.metrics, "created_at": self.created_at}
        if include_id:
            data["receipt_id"] = self.receipt_id
        return data

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target
