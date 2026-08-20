# Lesson 1 — Observation is cheap, causality is expensive

Attention weights, saliency, and probes are useful because they generate hypotheses. Their danger is rhetorical: they are visually persuasive even when they do not identify the computation the model uses.

## Exercise

Pick a binary prompt property and collect residual activations. Train a linear probe with `probe_with_controls`.

Then answer three separate questions:

1. Is the property decodable?
2. Does patching the activation change the output?
3. Does removing the candidate representation reduce the behavior?

If only (1) is true, you found information, not a mechanism.
