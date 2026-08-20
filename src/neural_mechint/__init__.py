"""Neural MechInt Lab: causal mechanistic interpretability tools and curricula."""

from .manifest import ExperimentManifest, hash_prompts
from .metrics import logit_difference, normalized_recovery, symmetric_kl
from .probes import mean_difference_direction, probe_with_controls

__all__ = [
    "ExperimentManifest",
    "hash_prompts",
    "logit_difference",
    "normalized_recovery",
    "symmetric_kl",
    "mean_difference_direction",
    "probe_with_controls",
]

__version__ = "0.2.0"
