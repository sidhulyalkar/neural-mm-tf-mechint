# Architecture

## Design principles

### One common causal surface

`HFCausalLMAdapter` exposes transformer block outputs as a portable residual-stream intervention surface. This gives the same patching and steering code a chance to work across Gemma, Llama, Qwen, GPT-2, GPT-NeoX, and related decoder-only Hugging Face models.

Architecture-specific attention-head or MLP-neuron experiments should be added as explicit adapters rather than hidden assumptions.

### Metrics are first-class objects

Interpretability methods receive a `score_fn(logits)` rather than hard-coding accuracy. This makes the same causal intervention usable for factual logit difference, refusal score, persona score, entropy, or another behavior.

### Search is not proof

Fast approximations such as probes and attribution patching are separated from exact intervention functions. This makes it harder to accidentally report a localization heuristic as causal evidence.

### Offline-testable core

Unit tests cover metrics, probe controls, circuit ranking, and attribution math without downloading model weights. Frontier-model tests belong in separate integration workflows because licensing, GPU availability, and model size vary by environment.

## Extension points

Future adapters can expose:

- attention head outputs;
- MLP pre/post activations;
- SAE feature activations;
- transcoder features;
- NNsight trace proxies;
- cross-model aligned feature spaces.
