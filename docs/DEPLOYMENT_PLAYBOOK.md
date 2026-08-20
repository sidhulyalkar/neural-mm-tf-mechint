# Deployment Playbook

## Phase A: discover

Run the belief-vs-compliance study, layer sweeps, activation patching, steering, and sparse feature analysis offline. Identify candidate signals and causal bottlenecks.

## Phase B: calibrate

Build held-out activation sets for each signal. Fit simple directions, choose thresholds from validation data, and freeze a versioned monitor bundle.

## Phase C: shadow

Run MechState in shadow mode. Log state receipts without changing model outputs. Estimate false-positive rate, latency, drift, and which requests would have been modified.

## Phase D: guarded canary

Enable intervention only for a small controlled traffic slice. Compare baseline and guarded responses, collect intervention gains, and monitor collateral behavior.

## Phase E: learn

Promote only successful intervention cases into MechTune. Compare targeted fine-tuning against standard LoRA. Re-run the full MechEval suite.

## Phase F: distill

When an intervention is stable and repeatedly useful, distill it into the model or a localized adapter. The goal is to reduce permanent inference overhead while preserving the desired computation.

## Rollback

Every runtime policy is explicit. Disable steering, return to shadow mode, or pin the prior probe/policy bundle without changing base weights.
