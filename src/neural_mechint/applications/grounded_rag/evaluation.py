from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingScore:
    mentions_evidence: bool
    rejects_false_claim: bool
    total: float


def score_response(response: str, *, evidence_terms: list[str], false_terms: list[str]) -> GroundingScore:
    text = response.lower()
    mentions_evidence = any(term.lower() in text for term in evidence_terms)
    rejects_false_claim = any(term.lower() in text for term in false_terms)
    return GroundingScore(mentions_evidence, rejects_false_claim, (float(mentions_evidence) + float(rejects_false_claim)) / 2.0)
