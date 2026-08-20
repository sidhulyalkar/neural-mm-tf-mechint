# Neural MechInt Lab

**A research-grade learning laboratory for mechanistic interpretability, from ground-truth toy circuits to causal studies of modern open language models.**

This repository is built around a simple rule:

> **A representation is not a mechanism until an intervention shows that the model uses it.**

The project therefore treats attention maps, probes, PCA plots, and feature visualizations as *hypothesis generators*. Mechanistic claims must climb a stricter evidence ladder: **observe → localize → intervene → transfer → test necessity/sufficiency → stress-test robustness**.

## Why this project exists

Mechanistic interpretability is moving from neuron inspection toward sparse feature dictionaries, transcoders, attribution graphs, causal mediation, activation patching, and scalable interventions. Neural MechInt Lab is designed as a hands-on package for learning that progression while producing experiments that could survive serious research scrutiny.

The repository has two complementary tracks:

1. **Controlled Mechanisms** — the original multimodal transformer remains useful as a small system where interventions are cheap and architectural assumptions can be checked directly.
2. **Frontier Model Laboratory** — a model-agnostic Hugging Face hook layer supports modern decoder-only LMs, with **Gemma 3** as the primary research family because Gemma Scope 2 provides unusually rich layer-wise sparse autoencoders and transcoders.

## Flagship research question: belief vs. compliance

When a language model knows a fact but a user confidently asserts a false answer and pressures the model to agree, what computation determines whether the model preserves the fact or complies?

The included experiment decomposes that question into falsifiable sub-hypotheses:

- **H1 — separability:** factual-state and social-pressure information become linearly separable at different depths.
- **H2 — causal bottleneck:** patching a neutral factual residual state into a pressured run restores the correct-answer logit difference in a localized layer band.
- **H3 — directional control:** a pressure direction extracted from held-out examples produces a monotonic steering dose-response.
- **H4 — partial disentanglement:** pressure and factual directions are not identical; one can alter compliance while preserving substantial factual competence.
- **H5 — sparse mediation:** a small subset of features/components can explain most of the recoverable behavioral effect.

The repository does **not** pre-declare these hypotheses as findings. The point is to make it easy to test, falsify, and refine them.

## Evidence ladder

| Level | Question | Methods in this repo | What you may claim |
|---|---|---|---|
| 0 | Does behavior change? | logit difference, KL, accuracy | behavioral association |
| 1 | Is information present? | linear probes, mean-difference directions | decodability |
| 2 | Where is it localized? | layer sweeps, attribution patching | candidate mechanism |
| 3 | Does changing it change behavior? | activation patching, ablation, steering | causal contribution |
| 4 | Is it necessary / sufficient? | circuit retention and knock-out curves | circuit-level faithfulness |
| 5 | Does it generalize? | held-out facts, prompt paraphrases, seeds, models | robust mechanism |

## Quick start

```bash
git clone https://github.com/sidhulyalkar/neural-mm-tf-mechint.git
cd neural-mm-tf-mechint
pip install -e ".[frontier,research,dev]"
```

Inspect a modern model:

```bash
mechint inspect --model google/gemma-3-270m-it
```

Run the flagship baseline:

```bash
mechint belief-compliance \
  --model google/gemma-3-270m-it \
  --output artifacts/belief_compliance_baseline.json
```

Then walk through the causal ladder:

```bash
python examples/01_mechanistic_ladder.py
```

> Some Gemma checkpoints require accepting the model license on Hugging Face and authenticating locally.

## Package map

```text
src/neural_mechint/
├── adapters.py                  # hookable Hugging Face causal-LM adapter
├── metrics.py                   # logit difference, KL, normalized recovery, bootstrap CIs
├── probes.py                    # controlled linear probes + representation directions
├── patching.py                  # activation patching + attribution-patching approximation
├── steering.py                  # causal steering dose-response experiments
├── circuits.py                  # component ranking + circuit faithfulness summaries
└── experiments/
    └── belief_compliance.py      # flagship factual-belief vs social-pressure study

docs/
├── METHODS.md                   # experiment design and statistical discipline
├── CLAIM_STANDARD.md            # what evidence supports which mechanistic claim
└── RESEARCH_PROGRAM.md          # frontier research roadmap

lessons/                         # progressive curriculum from probes to circuit tracing
examples/                        # executable research workflows
tests/                           # offline unit tests for core causal machinery
```

## Research workflow

A serious experiment should follow the same pattern:

1. **Define a behavioral metric before looking at activations.** Prefer a continuous metric such as correct-vs-incorrect logit difference.
2. **Create matched clean/counterfactual prompts.** Change one causal variable whenever possible.
3. **Cache residual states and run simple probes.** Include shuffled-label controls and held-out prompt families.
4. **Use fast localization only to nominate candidates.** Attribution patching is useful for search, not final proof.
5. **Run exact activation patching.** Measure normalized recovery relative to clean and corrupted baselines.
6. **Test interventions in both directions.** Positive and negative steering should produce a coherent dose-response.
7. **Measure collateral damage.** Track KL divergence and unrelated-task performance.
8. **Compress the mechanism.** Ask how little of the model is sufficient to reproduce the behavior.
9. **Break your own story.** Test paraphrases, new facts, positions, seeds, model sizes, and alternative metrics.

## Frontier extensions

The package is intentionally compatible with a broader ecosystem rather than attempting to reimplement every research stack. The next layer of experiments is designed to connect to:

- **Gemma Scope 2** SAEs, Matryoshka SAEs, skip-transcoders, and cross-layer transcoders.
- **Attribution graphs / circuit tracing** for sparse feature-level causal graphs.
- **NNsight** for intervention workflows and scaling to larger models or remote inference.
- **Mechanistic Interpretability Benchmark (MIB)** style faithfulness evaluation for circuit and causal-variable localization.
- **Model diffing / crosscoders** for comparing base vs. instruction-tuned or pre/post-finetuning mechanisms.

See [`docs/RESEARCH_PROGRAM.md`](docs/RESEARCH_PROGRAM.md) for the concrete study sequence.

## What makes a result interesting?

A colorful heatmap is not enough. A result becomes research-worthy when it contains a surprise that survives causal tests. Examples:

- a late-layer “compliance” direction that predicts agreement but is *not* the same direction that encodes truth;
- a small band of layers where neutral-state patching sharply restores factual answers under pressure;
- a feature that is weakly correlated observationally but has a large intervention effect;
- two model sizes that solve the same behavior using measurably different circuits;
- an instruction-tuned model that acquires a new social-pressure circuit while preserving the base model’s factual representation.

Those are the kinds of discoveries this package is designed to make reproducibly.

## Legacy multimodal sandbox

The original EEG/video/behavior/metadata transformer remains in the repository as a useful controlled teaching model. Its interpretability utilities are being treated as a sandbox rather than evidence about frontier LMs. This separation is intentional: toy models let us validate methods against known architecture; frontier models test whether those methods scale to representations we do not already understand.

## Reproducibility standard

Every publishable experiment should save:

- model identifier and revision;
- tokenizer revision;
- prompt dataset hash;
- random seeds;
- target metric definition;
- intervention location and coefficient;
- clean/corrupted/intervened scores;
- bootstrap confidence intervals;
- package and CUDA versions;
- generated artifact paths.

No cherry-picked prompt screenshots should be treated as primary evidence.

## Status

**v0.2 — Causal Research Laboratory**

The core experiment machinery, model adapter, controlled probes, patching, steering, circuit metrics, CI, and research curriculum are in place. The next research milestone is a complete Gemma 3 belief-vs-compliance report with sparse-feature mediation and cross-model replication.
