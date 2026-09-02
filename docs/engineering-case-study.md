# Engineering case study

## The problem

A neural time series captures only part of an experiment. Video-derived features, behavioral events, and task context can explain variation that a neural-only model cannot resolve. Combining them introduces several engineering questions: what constitutes one aligned example, how should heterogeneous signals become comparable representations, and how can we tell whether a trained model uses the inputs we intended?

This repository makes those questions concrete through a small, reproducible sequence-regression pipeline. The deliverable is an inspectable system, with a working training-to-dashboard path and explicit boundaries around its evidence.

## Design decisions and tradeoffs

| Decision | Rationale | Tradeoff / boundary |
|---|---|---|
| One shared `[batch, time]` grid | Gives encoders and the training loop a consistent interface | Timestamp synchronization and resampling must happen upstream |
| Different encoder per modality | Respects continuous neural signals, precomputed visual features, categorical events, and numeric context | The encoders are intentionally small reference implementations |
| Concatenate equal-width features | Establishes an understandable fusion baseline and a clean branch-removal boundary | Width becomes `4d`; there are no separate modality tokens |
| Explicit temporal attention blocks | Returns real per-head weights through the forward API, without fragile hooks or global fast-path changes | This block implementation must be tested and maintained |
| Synthetic target with known inputs | Makes it possible to detect broken learning and inspect controlled sensitivity without a data download | Synthetic signal recovery is not biological generalization |
| Constant and linear baselines | Measures whether extra model complexity is warranted for the task | The linear baseline is well matched to this generator and can win |
| Validation selects the checkpoint | Keeps test data out of model selection within a run | Repeated human tuning after looking at test results would still compromise the test set |
| Portable run artifacts | A reviewer can connect metrics, configuration, code hashes, and a reloadable model | This is a local experiment workflow, not an experiment database or deployment service |

## Follow one example through the system

1. **Generate or ingest.** `MultimodalDataset` returns a tuple of four aligned feature sequences and a target. For the demo, a seed keyed by experiment seed, split ID, and sample index makes each example stable across epochs.
2. **Batch.** `DataLoader` adds the batch dimension. `prepare_batch` separates model inputs from the target and moves tensors to the selected device. No hidden singleton dimension is required.
3. **Represent.** A temporal convolution maps neural channels into a learned representation. Linear projections handle video embeddings and metadata; an embedding table handles behavior IDs.
4. **Model.** The four representations are concatenated, time-position features are added, and a transformer produces one regression output per timestep.
5. **Select and evaluate.** The training loop selects the lowest-validation-MSE checkpoint. It then evaluates the test split and train-fitted baselines, with MSE weighted by the number of target elements.
6. **Inspect.** The dashboard reloads the checkpoint's own config. Temporal attention, branch removal, and head removal answer different questions and are labeled accordingly.

## How to review the engineering

Start with [the input boundary](../dataloaders.py) and [model forward pass](../model.py), then follow [training and artifact creation](../train.py). The [interpretability tests](../tests/test_interpretability.py) compare a head intervention against an independent attention calculation, verify that masked modalities stop affecting predictions, and check that model state is restored after failure.

The [pipeline tests](../tests/test_pipeline.py) exercise an actual CLI invocation, deterministic repeated training, and checkpoint reload. The [dashboard tests](../tests/test_dashboard.py) cover both an empty checkout and a saved run with a head intervention.

These are the concrete skills represented here: designing interfaces across heterogeneous data, developing modular neural models, connecting evaluation to model selection, debugging model-inspection semantics, and packaging the result so another person can run it.

## Status and attribution

The original project established the modality encoders, early-fusion transformer, simulator, and initial interpretability/dashboard modules. The 0.2 refresh makes that concept executable through deterministic data, a learnable task, validation/test evaluation, working artifacts, repaired inspection methods, tests, and this documentation.

The project is maintained by [Sidharth Hulyalkar](https://github.com/sidhulyalkar). Its synthetic results should be described as implementation evidence, not as experience validating this model on EEG, LFP, spikes, fMRI, or clinical data.
