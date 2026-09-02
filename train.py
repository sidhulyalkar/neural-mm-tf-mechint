"""Reproducible synthetic training, validation selection, and final test evaluation."""

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from baselines import fit_baselines, score_baselines
from configuration import load_config, validate_config
from dataloaders import get_loader, prepare_batch
from interpretability.ablation_study import modality_ablation
from interpretability.evaluation import evaluate
from model import MultimodalTransformer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_manifest():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (subprocess.SubprocessError, FileNotFoundError):
        commit, dirty = None, None
    files = sorted(
        list(root.glob("*.py"))
        + list((root / "encoders").glob("*.py"))
        + list((root / "interpretability").glob("*.py"))
    )
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"git_commit": commit, "working_tree_dirty": dirty, "source_sha256": hashes}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 checkpoint created by train.py")
    cfg = validate_config(checkpoint["config"])
    model = MultimodalTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def run_experiment(cfg, output, device="cpu", threads=1):
    validate_config(cfg)
    if threads < 1:
        raise ValueError("threads must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use --device cpu")
    output = Path(output)
    # A fresh run directory prevents accidental mixing or replacement of artifacts.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
    manifest = {
        "schema_version": 1,
        "data_kind": "synthetic",
        "seed": cfg["seed"],
        "split_ids": {"train": 0, "val": 1, "test": 2},
        "device": device,
        "threads": threads,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        **source_manifest(),
    }
    model = MultimodalTransformer(cfg).to(device)
    manifest["parameters"] = sum(p.numel() for p in model.parameters())
    train_loader = get_loader(cfg, "train")
    val_loader = get_loader(cfg, "val")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    baselines = fit_baselines(get_loader(cfg, "train", shuffle=False), cfg["data"]["behavioral_vocab"])
    initial = evaluate(model, val_loader, device)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            inputs, target = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(**inputs) - target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * target.numel()
            count += target.numel()
        validation = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": total / count, "val_mse": validation})
        if validation < best_loss:
            best_loss, best_epoch = validation, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d} | train MSE {total / count:.5f} | validation MSE {validation:.5f}")
    model.load_state_dict(best_state)
    model.eval()
    # The test set is first consumed after the checkpoint has been selected.
    test_loader = get_loader(cfg, "test")
    test_mse = evaluate(model, test_loader, device)
    baseline_scores = score_baselines(baselines, test_loader, cfg["data"]["behavioral_vocab"])
    metrics = {
        "schema_version": 1,
        "data_kind": "synthetic",
        "initial_val_mse": initial,
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "test_mse": test_mse,
        "test_baselines": baseline_scores,
        "history": history,
    }
    ablations = modality_ablation(model, test_loader, device)
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    (output / "config.yaml").write_text(config_text)
    manifest["config_sha256"] = hashlib.sha256(config_text.encode()).hexdigest()
    torch.save(
        {
            "schema_version": 1,
            "config": copy.deepcopy(cfg),
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_mse": best_loss,
        },
        output / "checkpoint.pt",
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "ablations.json", ablations)
    write_json(output / "baselines.json", baselines)
    write_json(output / "manifest.json", manifest)
    print(
        f"Test MSE {test_mse:.5f} | constant {baseline_scores['constant_mse']:.5f}"
        f" | ridge {baseline_scores['ridge_mse']:.5f}"
    )
    print(f"Artifacts: {output}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="runs/demo")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    run_experiment(load_config(args.config), args.output, args.device, args.threads)


if __name__ == "__main__":
    main()
