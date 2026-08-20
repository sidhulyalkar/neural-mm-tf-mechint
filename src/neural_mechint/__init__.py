"""Neural MechInt: mechanistic observability, causal control and feedback learning."""

from .manifest import ExperimentManifest, hash_prompts
from .metrics import logit_difference, normalized_recovery, symmetric_kl
from .probes import mean_difference_direction, probe_with_controls

__all__ = [
    "ExperimentManifest",
    "hash_prompts",
    "logit_difference",
    "mean_difference_direction",
    "normalized_recovery",
    "probe_with_controls",
    "symmetric_kl",
]

__version__ = "0.3.0"
