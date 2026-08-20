# MechTune

MechTune feeds causal interpretability findings back into model improvement.

## Mechanistically targeted data

Counterfactual generators vary the causal factor of interest, such as user pressure, while preserving the fact or trusted evidence. This creates sharper data than generic instruction augmentation.

## Representation objectives

The package exposes primitive losses for grounding preservation, representation separation, state matching, and intervention distillation. These are research tools, not automatic guarantees. Optimizing a probe can Goodhart the probe, so every representation objective must be re-evaluated with independent causal tests.

## Circuit-LoRA

Layer selection can be derived from measured intervention effects and collateral KL rather than defaulting to every attention/MLP block. Compare standard LoRA, random-layer LoRA, top-layer LoRA, and causally localized Circuit-LoRA.

## Intervention distillation

If an inference-time intervention reliably fixes a behavior, its output distribution and selected hidden states can serve as a teacher. The student is trained to reproduce the corrected trajectory without requiring the runtime intervention forever.

## Feedback admission rule

A monitor firing is insufficient. `FeedbackRecord` examples become training candidates only when the intervention measurably improves the target metric. This keeps the learning loop causal rather than self-referential.
