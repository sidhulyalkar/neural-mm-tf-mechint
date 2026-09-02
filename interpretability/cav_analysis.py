"""A binary linear concept probe, not a complete TCAV significance analysis."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_cav(activations, concepts):
    """Fit a concept separator on caller-provided [samples,features] activations.

    Fit and evaluate probes on independent data. This utility returns the raw
    classifier direction; it does not collect activations or test causal effects.
    """
    activations, concepts = np.asarray(activations), np.asarray(concepts)
    if activations.ndim != 2 or concepts.shape != (len(activations),):
        raise ValueError("Expected [samples, features] activations and [samples] labels")
    if set(np.unique(concepts)) != {0, 1}:
        raise ValueError("Concept labels must include both 0 and 1")
    classifier = LogisticRegression(random_state=0, max_iter=1000).fit(activations, concepts)
    return classifier.coef_[0]
