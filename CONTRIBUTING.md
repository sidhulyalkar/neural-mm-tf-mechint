# Development

Use Python 3.10+ and a virtual environment. From the repository root:

```bash
python -m pip install -e '.[dev,dashboard,concepts]'
python -m ruff check .
python -m pytest -q
python -m build
```

The tests run on synthetic CPU data. Dashboard tests use Streamlit's application-testing API and require the `dashboard` extra. CI installs that extra so these tests are not skipped.

For changes to model or data behavior, run the documented training command in a new output directory and reload the resulting checkpoint. Include the configuration, relevant validation, and limitations in the pull request. Update the input-contract documentation if shapes, dtypes, or split semantics change.

Keep generated checkpoints, downloaded data, and local run directories out of Git. Small example metrics may be checked in with their configuration, source/version manifest, and a precise description of what they support. Do not present simulated results as performance on biological recordings.

If you change an ablation or attention path, verify its numerical meaning and restoration behavior; a plausible visualization is not sufficient. Preserve the distinction between model sensitivity and biological causality.
