# Lesson 2 — Activation patching as a causal microscope

Activation patching performs an interchange intervention: run a clean example, transplant an internal state into a counterfactual run, and measure behavioral recovery.

## Checklist

- match prompts except for the variable of interest;
- choose a continuous output metric;
- align semantic positions carefully;
- record clean, corrupted, and patched scores;
- use normalized recovery;
- run both patch directions;
- include nearby-location controls.

## Challenge

Run `BeliefComplianceExperiment.patch_fact` on several facts. Do the same layers recover behavior across categories? If not, is the mechanism distributed, category-specific, or a token-alignment artifact?
