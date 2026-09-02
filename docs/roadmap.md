# Research and engineering roadmap

The implemented path is the synthetic CPU experiment and saved-run inspection. The following work is prospective.

| Next step | Why it matters | Completion evidence |
|---|---|---|
| Public dataset adapter | Moves beyond generated tensors to a scientific question | Documented source/license, sampling rates, clock mapping, feature provenance, and an executable adapter |
| Subject/session generalization | Tests transfer across the unit that matters | Frozen group-based splits created before windowing; training-only preprocessing; per-group metrics |
| Matched model comparisons | Determines whether complexity and each modality add value | Linear and unimodal models trained under the same split and selection policy; multi-seed uncertainty |
| Padding and missing-modality masks | Makes the contract useful for heterogeneous recordings | Masked loss/attention tests; variable-length batches; explicit missingness semantics |
| Fusion experiments | Tests early fusion against a substantive alternative | Integrated cross-attention or late fusion with controlled capacity and identical evaluation |
| Concept-probe evaluation | Turns a helper into a defensible interpretation experiment | Activation extraction, independent concept labels, held-out probe scores, and null/control tests |

Choose the dataset and scientific prediction target before increasing model size or adding infrastructure. No GPU run is necessary to reproduce the current software demonstration.
