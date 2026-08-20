# Frontier Research Program

The goal is not to collect techniques. The goal is to progressively tighten one scientific story until the model is forced to reveal which computation it actually uses.

## Study 1 — Belief vs. compliance atlas

**Question:** where does user-pressure information first become separable from factual information?

Experiments:

1. neutral vs pressured factual prompts across semantic categories;
2. per-layer linear-probe sweep with shuffled controls;
3. representational geometry: cosine similarity between pressure, correctness, sentiment, and instruction-following directions;
4. token-position sweep to distinguish user-claim encoding from answer-selection state;
5. cross-template holdout.

Deliverable: layer × position atlas with confidence intervals and negative controls.

## Study 2 — Causal bottleneck localization

**Question:** which internal states determine whether the model preserves truth under pressure?

Experiments:

1. exact residual activation patching for every layer;
2. bidirectional patches: neutral→pressured and pressured→neutral;
3. attribution patching to nominate MLP/attention subcomponents;
4. exact follow-up interventions on top candidates;
5. random-location and activation-norm controls.

Deliverable: a causal recovery map and a falsifiable bottleneck hypothesis.

## Study 3 — Pressure steering and collateral damage

**Question:** can the discovered representation control compliance without globally damaging the model?

Experiments:

1. difference-in-means direction on training facts;
2. held-out factual categories for evaluation;
3. positive/negative coefficient sweep;
4. unrelated factual benchmark and KL monitoring;
5. equal-norm random-direction baselines;
6. layer transfer: use a vector learned at L on neighboring layers where dimensions permit.

Deliverable: dose-response curves and specificity analysis.

## Study 4 — Sparse feature mediation with Gemma Scope 2

**Question:** is the causal effect distributed densely, or mediated by a small feature set?

Planned integration:

1. encode candidate-layer residual/MLP states using Gemma Scope 2 dictionaries;
2. rank SAE/transcoder features by activation difference and downstream attribution;
3. identify interpretable feature families associated with factual recall, social agreement, deference, uncertainty, and answer formatting;
4. intervene on individual and grouped features;
5. compare feature-level circuit recovery with dense residual patching.

A compelling result would be a small sparse feature set that recovers a large fraction of the dense intervention effect.

## Study 5 — Attribution graph

**Question:** how does pressure evidence flow into the final answer computation?

Build a sparse causal graph from source-token features through intermediate features to output logits. Use attribution graphs as a hypothesis generator, then validate the highest-impact paths by feature intervention.

The graph should distinguish:

- factual retrieval path;
- user-assertion / social-pressure path;
- conflict-resolution path;
- final answer-selection path.

## Study 6 — Base vs. instruction-tuned model diff

**Question:** does instruction tuning add a compliance mechanism, amplify an existing one, or rotate the representation into a new subspace?

Compare matched Gemma 3 base/instruction-tuned checkpoints using:

- per-layer activation differences;
- cross-model linear alignment;
- crosscoder/model-diff features where available;
- circuit overlap;
- intervention transferability.

This is particularly valuable because a mechanism that appears only after instruction tuning is more informative than a generic “sycophancy neuron.”

## Study 7 — Scale transition

Replicate the strongest findings across Gemma 3 sizes. Ask whether larger models:

- separate factual and social representations earlier;
- use sparser or more distributed circuits;
- resist pressure via a stronger factual path;
- develop new conflict-resolution features;
- show improved intervention specificity.

## Study 8 — Benchmark the interpretability method itself

Evaluate localization quality using MIB-style criteria:

- faithfulness to full-model behavior;
- circuit sparsity;
- causal-variable localization;
- random and heuristic baselines;
- robustness to metric choice.

The project becomes stronger when it studies not only the model, but also when its own interpretability tools fail.

## Publication-grade endpoint

A serious final report should contain:

1. a behavioral phenomenon with a controlled dataset;
2. a reproducible localization result;
3. exact causal interventions;
4. sparse circuit/feature mediation;
5. model-diff or scale replication;
6. negative controls and failure cases;
7. an explicit statement of what the evidence does *not* establish.
