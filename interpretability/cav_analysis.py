"""Controlled concept probes for the legacy multimodal teaching model."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compute_cav(activations, concepts, layer: str = "transformer"):
    """Fit a standardized linear concept classifier and return its normal vector.

    This function preserves the original API but should be interpreted as a *decodability*
    analysis. Use ``neural_mechint.probes.probe_with_controls`` for cross-validated metrics
    and shuffled-label controls before drawing conclusions.
    """

    x = np.asarray(activations, dtype=float)
    y = np.asarray(concepts, dtype=int)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("activations must be [samples, features] and concepts [samples]")
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    pipeline.fit(x, y)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    return classifier.coef_[0] / scaler.scale_
