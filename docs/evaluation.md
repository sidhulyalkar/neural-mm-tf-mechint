# Evaluation and evidence

## What the demo tests

The synthetic task checks an end-to-end software claim: four feature streams can be represented, fused, optimized against a learnable target, evaluated on unseen synthetic examples, saved, reloaded, and inspected.

Inputs are independently sampled. Let `n`, `v`, and `m` be the first neural, video, and metadata features at a timestep. Let `b` be the categorical event ID scaled from `[0, vocabulary − 1]` into `[-1, 1]`. The target is:

```text
y[t] = 0.90 n[t] + 0.65 v[t] + 0.50 b[t] + 0.35 m[t]
       + 0.25 n[t-1] + epsilon[t]
```

At the start of each sequence, the lag is zero. Noise is Gaussian with default standard deviation `0.05`. Neural, video, and metadata features are standard Gaussian draws; event IDs are uniformly sampled. The task is an intentionally transparent mathematical construction, not a physiological simulator.

## Evaluation protocol

- The default seed is 42, with 256 training, 64 validation, and 64 test sequences of 24 timesteps each.
- Each sample has a NumPy random stream derived from `[seed, split_id, sample_index]`. The three split IDs differ, and samples are stable when accessed again.
- Training uses AdamW, MSE, gradient clipping at norm 1.0, and 12 epochs. The lowest validation MSE selects the checkpoint.
- The test set is evaluated after checkpoint selection. Neither baseline fitting nor epoch selection uses test data.
- MSE aggregates squared error over all target elements, including a short final batch when dataset size is not divisible by batch size.
- Modality removals reuse exactly the same test samples and selected weights. Positive delta MSE indicates worse performance after removal.

The test-set ablations are descriptive diagnostics. Choosing subsequent hyperparameters using those results would make this test set part of development; a future final evaluation would need new held-out data.

## Baselines

The constant predictor is the training-target mean. The ridge model uses current continuous inputs, a scaled event ID, all one-step neural lags, and an intercept. Its fixed regularization coefficient is `0.001`; the intercept is unpenalized. It is fit on training examples only.

This baseline deliberately has access to the features needed to solve the generator. Because the underlying task is linear, ridge is expected to be strong. A transformer beating the constant predictor demonstrates successful learning; it does not establish a reason to prefer a transformer over the matched linear model.

Removing an encoded branch from a trained model changes its input distribution. It measures sensitivity under that intervention, not the accuracy of a separately trained unimodal baseline or causal importance in a biological system.

## Recorded example

[The CPU example](../examples/cpu-demo/README.md) contains measured outputs from the default configuration, together with the resolved config and source/version manifest. The manifest records the actual source hashes and whether the worktree was modified at execution; it does not imply that a dirty run was executed from an unmodified Git commit.

Reproduce it with a fresh output directory:

```bash
python train.py --config examples/cpu-demo/config.yaml --output runs/reproduction
```

Exact floating-point agreement is only expected with the same software, hardware behavior, and settings. A repeatability regression test verifies two identical runs in one CPU environment. There are no multi-seed confidence intervals or external benchmark scores yet.

## What would support a real modeling claim?

A useful next study would compare the transformer with matched linear, neural-only, behavior-only, and other modality subsets on a public recording dataset. It should report subject/session-separated splits, preprocessing fit boundaries, sampling/alignment assumptions, seed variation, performance per subject/session, and failure cases.

For interpretation, combine attention inspection with held-out interventions and suitable controls. For concept probes, separate probe fitting from evaluation and specify labels independently of model predictions. None of the current tools by themselves establish neural mechanisms.
