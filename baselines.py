"""Train-only constant and ridge baselines for the documented synthetic task."""

import numpy as np


def baseline_arrays(loader, vocabulary):
    features, targets = [], []
    for neural, video, behavior, meta, target in loader:
        neural, video, meta = (x.numpy() for x in (neural, video, meta))
        events = 2.0 * behavior.numpy()[..., None] / (vocabulary - 1) - 1.0
        lag = np.zeros_like(neural)
        lag[:, 1:] = neural[:, :-1]
        # The lag is computed inside each sequence, never across split boundaries.
        x = np.concatenate([neural, video, events, meta, lag], axis=-1)
        features.append(x.reshape(-1, x.shape[-1]))
        targets.append(target.numpy().reshape(-1))
    return np.concatenate(features).astype("float64"), np.concatenate(targets).astype("float64")


def fit_baselines(loader, vocabulary):
    x, y = baseline_arrays(loader, vocabulary)
    x = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x.shape[1]) * 1e-3
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {
        "constant": float(y.mean()),
        "ridge_weights": weights.tolist(),
        "ridge_alpha": 1e-3,
        "features": "current inputs, scaled event ID, one-step neural lag",
    }


def score_baselines(fitted, loader, vocabulary):
    x, y = baseline_arrays(loader, vocabulary)
    x = np.column_stack([np.ones(len(x)), x])
    prediction = x @ np.asarray(fitted["ridge_weights"])
    return {
        "constant_mse": float(np.mean((y - fitted["constant"]) ** 2)),
        "ridge_mse": float(np.mean((y - prediction) ** 2)),
    }
