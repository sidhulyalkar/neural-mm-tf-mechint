# MechState

MechState models an LLM forward pass as an interpretable state trajectory across depth.

For layer `l`, a deployment can define a compact state vector such as:

`S_l = [retrieval_grounding, factual_evidence, uncertainty, compliance_pressure, answer_commitment]`

The coordinates are not claimed to be ontologically privileged. They are operational variables whose validity must be tested.

## Calibration protocol

1. Define positive and negative examples without leakage.
2. Cache the same residual position and layer family.
3. Fit a transparent direction first.
4. Evaluate on held-out prompts, paraphrases, topics, and seeds.
5. Compare with shuffled-label and equal-norm random directions.
6. Demonstrate that moving along the direction changes the predicted behavior.
7. Check specificity and collateral effects.
8. Version the probe, thresholds, model hash, prompt family, and calibration dataset.

`fit_mean_difference_probe` intentionally provides a simple baseline. A more complicated classifier may improve prediction while making the state variable less scientifically trustworthy.

## State receipts

Every guarded request may emit a receipt containing the trajectory, decision, reasons, intervention metadata, and metrics. Prompt text is not required. The reference deployment config defaults to prompt IDs rather than raw content.
