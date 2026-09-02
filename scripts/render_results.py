"""Render a compact README figure from a saved run's measured metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/cpu-results.png"))
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    ablations = json.loads((args.run / "ablations.json").read_text())
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fig.set_facecolor("#f8fafc")
    fig.suptitle("Multimodal pipeline · synthetic CPU evaluation", fontsize=17, fontweight="bold")
    history = metrics["history"]
    epochs = [row["epoch"] for row in history]
    axes[0].semilogy(epochs, [row["train_mse"] for row in history], label="Training", color="#64748b")
    axes[0].semilogy(
        epochs, [row["val_mse"] for row in history], label="Validation", color="#007f86", linewidth=2.5
    )
    axes[0].axvline(metrics["best_epoch"], color="#007f86", linestyle=":", alpha=0.7)
    axes[0].set(title="Validation selects the checkpoint", xlabel="Epoch", ylabel="MSE · log scale")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.15)
    labels = ["Neural", "Video", "Behavior", "Metadata"]
    values = [ablations["ablations"][key]["delta_mse"] for key in ("neural", "video", "behavior", "meta")]
    bars = axes[1].barh(labels[::-1], values[::-1], color=["#8b75b8", "#d99a38", "#4689b5", "#007f86"])
    axes[1].bar_label(bars, fmt="%.3f", padding=5)
    axes[1].set(title="Effect of removing an encoded modality", xlabel="Change in test MSE · no retraining")
    axes[1].set_xlim(min(0, min(values) * 1.25), max(0.01, max(values) * 1.25))
    axes[1].grid(axis="x", alpha=0.15)
    fig.supxlabel(
        f"Known synthetic target · {config['data']['train_size']} train / "
        f"{config['data']['val_size']} validation / {config['data']['test_size']} test sequences "
        f"· seed {config['seed']}",
        fontsize=10,
        color="#475569",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
