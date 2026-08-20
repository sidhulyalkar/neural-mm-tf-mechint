from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    roc_auc: float
    shuffled_accuracy: float
    shuffled_roc_auc: float
    n_samples: int
    n_features: int


def mean_difference_direction(positive: np.ndarray, negative: np.ndarray, unit_norm: bool = True) -> np.ndarray:
    """Difference-in-means representation direction, optionally normalized."""

    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1]:
        raise ValueError("positive and negative must be [samples, features] with matching features")
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    if unit_norm:
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
    return direction


def _cv_predictions(x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    min_class = int(np.bincount(y).min())
    n_splits = min(folds, min_class)
    if n_splits < 2:
        raise ValueError("need at least two examples in every class")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    return cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]


def probe_with_controls(
    activations: np.ndarray,
    labels: np.ndarray | list[int],
    folds: int = 5,
    seed: int = 0,
) -> ProbeResult:
    """Cross-validated linear probe with a shuffled-label negative control.

    A probe is evidence that information is *decodable*, not that it is used by the model.
    This helper intentionally pairs the probe with a sanity-control baseline.
    """

    x = np.asarray(activations, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("activations must be [samples, features] and labels [samples]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")

    probs = _cv_predictions(x, y, folds, seed)
    pred = (probs >= 0.5).astype(int)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(y)
    shuffled_probs = _cv_predictions(x, shuffled, folds, seed + 1)
    shuffled_pred = (shuffled_probs >= 0.5).astype(int)

    return ProbeResult(
        accuracy=float(accuracy_score(y, pred)),
        roc_auc=float(roc_auc_score(y, probs)),
        shuffled_accuracy=float(accuracy_score(shuffled, shuffled_pred)),
        shuffled_roc_auc=float(roc_auc_score(shuffled, shuffled_probs)),
        n_samples=int(x.shape[0]),
        n_features=int(x.shape[1]),
    )
