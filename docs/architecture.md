# Architecture and data contracts

## Input contract

The model boundary accepts **already aligned, fixed-length sequences**. All four inputs share batch size `B` and sequence length `T`.

| Name | Batched shape | Demo dtype | Meaning |
|---|---|---|---|
| `neural` | `[B, T, C]` | `float32` | Continuous neural features; demo values are simulated |
| `video` | `[B, T, V]` | `float32` | Precomputed video embeddings, not raw image frames |
| `behavior` | `[B, T]` | `int64` | Categorical IDs in `[0, behavioral_vocab)` |
| `meta` | `[B, T, M]` | `float32` | Numeric task/context features available at inference |
| `target` | `[B, T, 1]` | `float32` | Per-timestep regression label, passed to the loss only |

A dataset item omits `B`; batching adds it. Task metadata that is constant over a sequence must be explicitly broadcast over time before it enters the model. No timestamp units or physiological sampling rate are assigned to the synthetic grid.

### Adapting real data

A real adapter should establish a shared clock, align event timestamps, resample continuous signals with an appropriate anti-aliasing policy, and document how video windows map to each timestep. Split by the intended unit of generalization—typically subject, session, or recording—**before** producing overlapping windows. Fit normalization and any learned feature extraction only on training data, unless using a documented frozen external encoder.

Decide which signals and metadata are actually available at inference. Exclude labels, future outcomes, and target-derived task annotations from the features. Record the dataset version, processing choices, and split membership alongside results.

These are integration requirements, not existing loader capabilities. The repository does not contain a biological dataset adapter, clock synchronization, raw video decoding, padding masks, or missing-modality handling.

## Encoders and fusion

| Component | Transformation | Design note |
|---|---|---|
| `NeuralEncoder` | `[B,T,C] → [B,C,T] → Conv1d(C,d,k=3) → [B,T,d]` | Centered convolution captures a short local context |
| `VideoEncoder` | `Linear(V,d)` | Assumes a feature extractor has already processed frames |
| `BehaviorEncoder` | `Embedding(vocabulary,d)` | Learns a representation per event category |
| `MetadataEncoder` | `Linear(M,d)` | Projects numeric task context |
| `early_fusion` | Concatenate on the feature axis | Produces `[B,T,4d]` |

The default config uses `d=16`, so each temporal token is 64-wide. Sinusoidal positions encode absolute time indices within the window. Each transformer block applies multihead self-attention, residual/dropout/layer normalization, then a two-layer GELU feed-forward network with its own residual and normalization. A final linear head returns `[B,T,1]`.

Both the neural convolution and temporal attention can see future positions in the supplied window. This is an **offline model**. A streaming decoder would need causal convolutions, attention masks, a latency contract, and separate validation.

`CrossModalAttention` remains a standalone `[T,B,d]` primitive in `fusion.py`; it is not the active fusion path. No cross-attention performance comparison is claimed.

## Model inspection boundaries

| Tool | What it measures | What it does not establish |
|---|---|---|
| Temporal attention | Attention probability from each query timestep to each key timestep, for each head | Causal influence, modality attribution, or biological connectivity |
| Encoded-modality removal | MSE change when a complete encoded branch is zeroed, on the same examples | Performance of a separately trained unimodal model; resilience to natural missingness |
| Attention-head removal | MSE after removing a selected head's contribution to the output projection | A unique circuit or a causal mechanism in the brain |
| Binary concept probe | A linear separating direction for supplied activations and binary concept labels | Full TCAV analysis, statistical significance, or an automatically discovered concept |

The attention API requests `need_weights=True` and `average_attn_weights=False` from [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), producing `[B,H,T,T]`. The helper returns the first example as `[H,T,T]`, or one head as `[T,T]`.

For head removal, heads occupy contiguous input slices of the final attention output projection. The implementation zeros the corresponding **columns** of `out_proj.weight`, then restores all weights in a `finally` block. An independent attention calculation in the tests verifies this operation. It temporarily mutates the model, so callers must not share that model with concurrent inference. The dashboard creates its own model for each rerun.

## Files and responsibilities

| Path | Responsibility |
|---|---|
| `configuration.py` | YAML loading and experiment validation |
| `data_simulation.py`, `dataloaders.py` | Synthetic task and stable split-aware batches |
| `encoders/`, `fusion.py`, `model.py` | Representations and temporal regression |
| `baselines.py` | Train-fitted constant and ridge models |
| `train.py` | CLI, optimization, checkpoint selection, artifacts |
| `interpretability/` | Evaluation, attention, ablations, optional concept probe |
| `dashboard.py` | Saved-run inspection |
| `configs/`, `tests/`, `.github/workflows/` | Configuration and repeatable validation |

## Compatibility

Version 0.2 keeps the four-input model call and encoder names, but introduces explicit transformer blocks, time-position encoding, and a versioned checkpoint envelope. Legacy state dictionaries from the initial scaffold are not compatible; retrain with the new configuration. `ablate_heads` returns absolute post-ablation MSE; subtract a paired baseline to obtain the change. `compute_cav` takes activations and labels, without the old unused `layer` argument.
