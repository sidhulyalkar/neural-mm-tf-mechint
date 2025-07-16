# interpretability/cav_analysis.py
"""
This module provides functions to compute concept activation vectors.

Functions:
    compute_cav(activations, concepts): Computes concept activation vector.
"""
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_cav(activations, concepts, layer='transformer'):
    """
    Given activations and binary labels for a concept,
    fit linear classifier and return concept activation vector.
    """
    # activations: [samples, features]
    clf = LogisticRegression().fit(activations, concepts)
    return clf.coef_[0]
