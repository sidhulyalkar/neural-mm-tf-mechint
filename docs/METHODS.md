# Methods

## 1. Behavioral metrics first

Mechanistic studies start with a scalar behavior. For two candidate answers, this project uses

\[
\Delta \ell = \ell(\text{correct}) - \ell(\text{incorrect})
\]

at the answer position. Continuous logit differences are preferable to binary accuracy because they preserve effect size even when an intervention does not cross the argmax boundary.

For broad behavioral drift, use symmetric KL divergence over the output distribution. This catches interventions that improve the target metric only by damaging the model globally.

## 2. Clean/corrupted counterfactuals

The flagship experiment uses paired prompts:

- **clean / neutral**: ask for a fact directly;
- **corrupted / pressured**: inject a confident false claim and request agreement.

The desired pair changes the social-pressure variable while preserving the underlying factual question.

## 3. Activation patching

For block \(L\), let \(h_L^c\) be the clean residual state and \(h_L^x\) the corrupted state. We replace the corrupted state at a selected position with the clean state and continue the forward pass.

The normalized recovery is

\[
R = \frac{s_{patch} - s_{corrupt}}{s_{clean} - s_{corrupt}}.
\]

Interpretation:

- \(R=0\): no recovery;
- \(R=1\): full clean behavior recovered;
- \(R<0\): intervention moves behavior farther from clean;
- \(R>1\): overshoot.

Do not clip recovery to [0, 1]. Overshoot contains mechanistic information.

## 4. Attribution patching

Attribution patching approximates the effect of replacing an activation using

\[
(h^c - h^x) \odot \nabla_{h^x} s.
\]

It is excellent for searching many candidate nodes or edges cheaply. Because it is a first-order approximation, top candidates must be validated with exact interventions.

## 5. Probes

A linear probe asks whether a variable can be decoded from an activation. The default pipeline uses standardization, cross-validation, class balancing, and a shuffled-label control.

The project deliberately reports probe performance separately from causal intervention results. A perfect probe can coexist with zero causal use.

## 6. Representation directions and steering

For positive and negative activation sets,

\[
v = \frac{\mu_+ - \mu_-}{\|\mu_+ - \mu_-\|}.
\]

A causal steering test adds \(\alpha v\) to the residual stream. Strong evidence requires a coherent dose-response over multiple positive and negative coefficients, not one hand-picked steering value.

### Steering controls

- equal-norm random directions;
- unrelated semantic directions;
- multiple layers;
- positive and negative coefficients;
- KL divergence / capability preservation;
- held-out prompts.

## 7. Circuit faithfulness

Circuit discovery is evaluated as a curve, not a single threshold. Rank candidate components, retain increasingly large fractions, and measure how much of the original model behavior is recovered. Also perform the complementary necessity experiment by removing the discovered circuit.

Random size-matched component sets provide the null baseline.

## 8. Statistical discipline

For repeated prompt-level effects, report means with bootstrap confidence intervals. For directional experiments, report paired per-example changes. Avoid treating thousands of tokens from the same small prompt set as independent samples.

## 9. Artifact discipline

Each experiment should emit machine-readable JSON/CSV plus figures. Every artifact should include model revision, tokenizer revision, prompt hash, seed, layer, position, intervention coefficient, and metric definition.
