# Lesson 0 — Start where the answer is knowable

Before interpreting a billion-parameter model, learn to catch yourself making invalid causal claims on a small one.

## Goals

- distinguish architecture knowledge from empirical evidence;
- understand residual streams, attention, MLPs, and output projections;
- verify that an ablation removes the component you think it removes;
- compare observation, ablation, and interchange intervention.

## Exercise

Use the repository's multimodal transformer. Construct a synthetic target that depends on exactly one modality at one time step. Train until the dependency is reliable. Then test:

1. input-modality ablation;
2. residual activation patching between examples;
3. head/output projection ablation;
4. a linear probe from every layer.

Write down which methods correctly recover the planted dependency and which produce convincing-but-wrong stories.
