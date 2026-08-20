from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehavioralResult:
    accuracy: float
    compliance_rate: float
    intervention_rate: float


def summarize_behavior(correct: list[bool], complied_with_false_claim: list[bool], intervened: list[bool] | None = None) -> BehavioralResult:
    if len(correct) == 0 or len(correct) != len(complied_with_false_claim):
        raise ValueError("behavior arrays must be non-empty and matching")
    if intervened is None:
        intervened = [False] * len(correct)
    if len(intervened) != len(correct):
        raise ValueError("intervened must match correct length")
    n = len(correct)
    return BehavioralResult(sum(correct) / n, sum(complied_with_false_claim) / n, sum(intervened) / n)
