from pathlib import Path

import pytest
import torch

from configuration import load_config
from train import run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    config = load_config(ROOT / "configs/config.yaml")
    config["model"].update(d_model=4, n_heads=2, num_layers=1, dropout=0.0)
    config["data"].update(seq_length=6, train_size=16, val_size=7, test_size=9)
    config["train"].update(batch_size=4, epochs=2)
    torch.set_num_threads(1)
    return config


@pytest.fixture
def trained_run(cfg, tmp_path):
    output = tmp_path / "run"
    run_experiment(cfg, output)
    return output
