# Evaluation Protocol

A custom deployment should be evaluated at four levels.

## Behavioral

Task accuracy, false-compliance rate, calibration, abstention quality, and domain-specific metrics.

## Mechanistic state

Monitor recall/precision for known failure cases, state-trajectory stability, crossover depth, probe calibration, and out-of-distribution degradation.

## Causal control

Intervention success, dose response, necessity/sufficiency curves, random-direction controls, equal-norm controls, and collateral KL on unrelated prompts.

## Operational

Intervention rate, latency overhead, memory overhead, receipt storage cost, and escalation rate.

`DeploymentScorecard` deliberately combines these categories. A guard that fixes every adversarial prompt by intervening on every request is not a successful controller.

## Release gate

Do not enable an intervention by default until it passes held-out behavioral improvement, causal specificity, acceptable collateral KL, latency budget, OOD stress tests, model-version compatibility, and receipt/audit review.
