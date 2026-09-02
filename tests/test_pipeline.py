import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from configuration import validate_config
from data_simulation import simulate_batch
from dataloaders import MultimodalDataset, get_loader, prepare_batch
from interpretability.evaluation import evaluate
from model import MultimodalTransformer
from train import load_checkpoint, run_experiment


def test_dataset_contract_and_split_independence(cfg):
    train = MultimodalDataset(cfg, split="train")
    val = MultimodalDataset(cfg, split="val")
    for first, repeated in zip(train[0], train[0]):
        assert torch.equal(first, repeated)
    assert not torch.equal(train[0][0], val[0][0])
    assert not torch.equal(train[0][0], train[1][0])
    inputs, target = prepare_batch(next(iter(get_loader(cfg))))
    assert inputs["neural"].shape == (4, 6, 8)
    assert inputs["behavior"].shape == (4, 6)
    assert inputs["behavior"].dtype == torch.int64
    assert target.shape == (4, 6, 1)
    assert target.dtype == torch.float32


def test_synthetic_target_matches_documented_relationship(cfg):
    cfg["data"]["noise_std"] = 0
    neural, video, events, meta, target = simulate_batch(cfg, 2, np.random.default_rng(2))
    scaled = 2 * events / (cfg["data"]["behavioral_vocab"] - 1) - 1
    expected = 0.9 * neural[..., 0] + 0.65 * video[..., 0] + 0.5 * scaled + 0.35 * meta[..., 0]
    expected[:, 1:] += 0.25 * neural[:, :-1, 0]
    np.testing.assert_allclose(target[..., 0], expected, atol=1e-6)


def test_all_encoders_receive_gradients_and_inputs_are_checked(cfg):
    model = MultimodalTransformer(cfg)
    inputs, target = prepare_batch(next(iter(get_loader(cfg))))
    predictions = model(**inputs)
    assert predictions.shape == target.shape
    (predictions - target).square().mean().backward()
    for encoder in (model.neural_enc, model.video_enc, model.behav_enc, model.meta_enc):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    with pytest.raises(ValueError, match="aligned"):
        model(**{**inputs, "video": inputs["video"][:, :-1]})
    with pytest.raises(ValueError, match="int64"):
        model(**{**inputs, "behavior": inputs["behavior"].float()})
    with pytest.raises(ValueError, match="vocabulary"):
        model(**{**inputs, "behavior": torch.full_like(inputs["behavior"], 999)})
    with pytest.raises(ValueError, match="Unknown modalities"):
        model(**inputs, ablate_modalities=["eeg"])


def test_config_rejects_invalid_head_width(cfg):
    cfg["model"]["n_heads"] = 3
    with pytest.raises(ValueError, match="divisible"):
        validate_config(cfg)


def test_saved_checkpoint_selection_and_reproducibility(cfg, trained_run, tmp_path):
    metrics = json.loads((trained_run / "metrics.json").read_text())
    assert metrics["best_val_mse"] == min(row["val_mse"] for row in metrics["history"])
    model, checkpoint = load_checkpoint(trained_run / "checkpoint.pt")
    assert checkpoint["config"] == cfg
    assert evaluate(model, get_loader(cfg, "test")) == metrics["test_mse"]
    repeated = run_experiment(cfg, tmp_path / "repeat")
    assert repeated == metrics
    assert metrics["test_baselines"]["ridge_mse"] < metrics["test_baselines"]["constant_mse"]
    ablations = json.loads((trained_run / "ablations.json").read_text())
    assert ablations["baseline_mse"] == metrics["test_mse"]
    assert set(ablations["ablations"]) == {"neural", "video", "behavior", "meta"}
    with pytest.raises(ValueError, match="not empty"):
        run_experiment(cfg, trained_run)


def test_cli_honors_configuration(cfg, tmp_path):
    cfg["seed"] = 123
    cfg["train"]["epochs"] = 1
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "cli"
    script = Path(__file__).resolve().parents[1] / "train.py"
    subprocess.run(
        [sys.executable, str(script), "--config", str(path), "--output", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    _, checkpoint = load_checkpoint(output / "checkpoint.pt")
    assert checkpoint["config"] == cfg
    assert checkpoint["best_epoch"] == 1
