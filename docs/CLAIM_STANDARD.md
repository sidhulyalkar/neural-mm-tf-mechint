# Mechanistic Claim Standard

Mechanistic interpretability fails when the strength of the prose outruns the strength of the intervention. This project uses explicit claim tiers.

## Tier A — observation

Examples: attention heatmaps, neuron activation plots, PCA/UMAP, feature dashboards.

Allowed claim: **"this component is associated with the behavior."**

Not allowed: **"this component causes the behavior."**

## Tier B — decodability

Evidence: held-out linear probe, representation direction, classifier built from activations.

Required controls:

- held-out examples;
- shuffled-label baseline;
- class balance reporting;
- prompt/template split where possible;
- multiple seeds.

Allowed claim: **"information about X is decodable from layer L."**

A probe can exploit information the model never uses downstream, so decodability is not mechanism.

## Tier C — causal localization

Evidence: exact activation patching, interchange intervention, or component ablation on a matched clean/corrupted pair.

Required controls:

- clean and corrupted behavioral baselines;
- normalized recovery;
- patch direction reversal where meaningful;
- nearby-layer negative controls;
- multiple examples.

Allowed claim: **"the activation at location L causally contributes to the behavior under this intervention."**

## Tier D — circuit claim

Evidence: a sparse set of nodes/edges/components that is both necessary and approximately sufficient.

Required controls:

- sufficiency curve: retain only the circuit;
- necessity curve: remove only the circuit;
- circuit-size vs behavior trade-off;
- random-component baselines matched for size;
- faithfulness evaluated against the full model, not merely ground-truth labels.

Allowed claim: **"this circuit captures a substantial fraction of the model computation for behavior X."**

## Tier E — robust mechanism

Evidence: Tier D plus replication.

Replication axes should include at least three of:

- prompt paraphrases;
- unseen semantic items;
- sequence positions;
- random seeds;
- model sizes;
- base vs instruction-tuned checkpoints;
- alternative behavioral metrics;
- alternative localization methods.

Allowed claim: **"this appears to be a reusable mechanism rather than a prompt-local intervention artifact."**

## Red-team questions for every result

1. Could tokenization explain the effect?
2. Is the intervention merely changing activation norm?
3. Does a random direction of equal norm do the same thing?
4. Does the effect survive held-out semantic content?
5. Does the effect disappear if the target metric changes slightly?
6. Are we patching aligned semantic positions, or only matching token indices?
7. Does the intervention destroy unrelated capabilities?
8. Is the discovered circuit faithful to the *model's behavior* even when the model is wrong?
9. Could probe leakage or template artifacts explain decodability?
10. Does the causal effect replicate under a second method?
