from __future__ import annotations

from dataclasses import dataclass

import torch

from .estimators import LinearSignalProbe


@dataclass(frozen=True)
class ProbeCalibrationReport:
    name: str
    train_accuracy: float
    positive_mean_score: float
    negative_mean_score: float
    direction_norm: float
    n_positive: int
    n_negative: int


def fit_mean_difference_probe(name: str, positive: torch.Tensor, negative: torch.Tensor, *, normalize: bool = True, source: str = "mean-difference") -> tuple[LinearSignalProbe, ProbeCalibrationReport]:
    """Fit a transparent deployment probe from positive/negative activation examples."""
    if positive.ndim != 2 or negative.ndim != 2 or positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative must be [N,D] tensors with matching D")
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("positive and negative must be non-empty")
    pos_mean = positive.float().mean(dim=0)
    neg_mean = negative.float().mean(dim=0)
    direction = pos_mean - neg_mean
    norm = direction.norm()
    if float(norm) == 0.0:
        raise ValueError("class means are identical; no mean-difference direction exists")
    if normalize:
        direction = direction / norm
    pos_projection = positive.float() @ direction
    neg_projection = negative.float() @ direction
    midpoint = 0.5 * (pos_projection.mean() + neg_projection.mean())
    probe = LinearSignalProbe(name=name, direction=direction, bias=-float(midpoint), source=source)
    pos_scores = torch.sigmoid(pos_projection - midpoint)
    neg_scores = torch.sigmoid(neg_projection - midpoint)
    correct = torch.cat([pos_scores >= 0.5, neg_scores < 0.5]).float().mean()
    report = ProbeCalibrationReport(name, float(correct), float(pos_scores.mean()), float(neg_scores.mean()), float(direction.norm()), len(positive), len(negative))
    return probe, report
