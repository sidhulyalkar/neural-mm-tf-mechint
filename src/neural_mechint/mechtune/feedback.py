from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackRecord:
    """Privacy-minimal record connecting a mechanistic trigger to intervention outcome."""
    prompt_id: str
    decision: str
    baseline_metric: float
    intervened_metric: float
    risk_score: float
    trigger_layers: tuple[int, ...]
    trigger_signals: tuple[str, ...]

    @property
    def intervention_gain(self) -> float:
        return self.intervened_metric - self.baseline_metric


@dataclass(frozen=True)
class TrainingCandidate:
    prompt_id: str
    priority: float
    target_layers: tuple[int, ...]
    reason: str


def select_training_candidates(records: list[FeedbackRecord], *, min_gain: float = 0.05, top_k: int | None = None) -> list[TrainingCandidate]:
    """Prioritize examples where a causal intervention actually improved the target metric."""
    candidates = []
    for record in records:
        if record.intervention_gain < min_gain:
            continue
        priority = record.intervention_gain * max(record.risk_score, 1e-6)
        candidates.append(TrainingCandidate(record.prompt_id, priority, tuple(sorted(set(record.trigger_layers))), f"intervention gain={record.intervention_gain:.3f}, risk={record.risk_score:.3f}, signals={','.join(record.trigger_signals)}"))
    candidates.sort(key=lambda row: row.priority, reverse=True)
    return candidates if top_k is None else candidates[:top_k]
