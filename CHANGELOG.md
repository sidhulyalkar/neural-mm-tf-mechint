# Changelog

## 0.2.0 — Reproducible multimodal pipeline

- Replace the independent random target with a documented multimodal synthetic task and stable train/validation/test samples.
- Honor CLI configuration; add validation-based selection, held-out evaluation, constant/ridge baselines, checkpoints, and provenance artifacts.
- Add time-position encoding and explicit transformer blocks that return real per-head attention weights.
- Correct head ablation to remove output-projection input columns, restore weights after failure, and separate targets from model inputs.
- Add encoded-modality ablations and a working saved-run Streamlit dashboard.
- Add installable project metadata, CPU CI, regression tests, and an architecture/evaluation case study.

The model checkpoint format and temporal blocks changed. Retrain instead of loading a legacy state dictionary. The project remains a synthetic research prototype.
