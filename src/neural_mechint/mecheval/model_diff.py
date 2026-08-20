from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass
class MechanisticModelDiff:
    behavioral: list[MetricDelta]
    state: list[MetricDelta]
    causal: list[MetricDelta]

    def largest_changes(self, *, top_k: int = 5) -> list[MetricDelta]:
        rows = [*self.behavioral, *self.state, *self.causal]
        return sorted(rows, key=lambda row: abs(row.delta), reverse=True)[:top_k]

    def to_dict(self) -> dict[str, object]:
        def serialize(rows: list[MetricDelta]) -> list[dict[str, float | str]]:
            return [{**asdict(row), "delta": row.delta} for row in rows]
        return {"behavioral": serialize(self.behavioral), "state": serialize(self.state), "causal": serialize(self.causal)}
