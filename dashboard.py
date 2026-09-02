"""Inspect a saved synthetic run: curves, predictions, temporal attention, ablations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from dataloaders import get_loader, prepare_batch
from interpretability.ablation_study import ablate_heads
from interpretability.attention_analysis import extract_attention_maps
from train import load_checkpoint


def main():
    st.set_page_config(page_title="Multimodal Neural Modeling", layout="wide")
    st.title("Multimodal Neural Modeling")
    st.caption("Neural features · video embeddings · behavioral events · task metadata")
    st.info(
        "Synthetic research demo. Metrics describe the generated task, not biological decoding performance."
    )
    run_dir = Path(st.sidebar.text_input("Run directory", "runs/demo"))
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        st.write("Train the CPU example to create a checkpoint and evaluation artifacts:")
        st.code("python train.py --config configs/config.yaml --output runs/demo")
        return
    try:
        # Each rerun owns its model; temporary interventions never mutate a shared cache.
        model, checkpoint = load_checkpoint(checkpoint_path)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        ablations = json.loads((run_dir / "ablations.json").read_text())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        st.error(f"Cannot load this run: {error}")
        return
    cfg = checkpoint["config"]
    cols = st.columns(3)
    cols[0].metric("Transformer test MSE", f"{metrics['test_mse']:.4f}")
    cols[1].metric("Constant baseline MSE", f"{metrics['test_baselines']['constant_mse']:.4f}")
    cols[2].metric("Linear ridge MSE", f"{metrics['test_baselines']['ridge_mse']:.4f}")
    st.caption(
        f"Checkpoint selected on validation data at epoch {checkpoint['best_epoch']}. Lower MSE is better."
    )
    st.subheader("Training and validation")
    st.line_chart(pd.DataFrame(metrics["history"]).set_index("epoch"))
    inputs, target = prepare_batch(next(iter(get_loader(cfg, "test"))))
    with torch.no_grad():
        predictions = model(**inputs)
    st.subheader("Held-out sequence")
    st.line_chart(
        pd.DataFrame({"target": target[0, :, 0].numpy(), "prediction": predictions[0, :, 0].numpy()})
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Temporal attention")
        layer = st.selectbox("Layer", range(cfg["model"]["num_layers"]))
        head = st.selectbox("Head", range(cfg["model"]["n_heads"]))
        attention = extract_attention_maps(model, inputs, layer, head)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(attention, cmap="viridis", aspect="auto")
        ax.set(xlabel="Key timestep", ylabel="Query timestep")
        fig.colorbar(image, ax=ax, label="Attention weight")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "These weights connect time positions after fusion; they are not modality attribution or causal evidence."
        )
    with right:
        st.subheader("Modality removal")
        table = pd.DataFrame.from_dict(ablations["ablations"], orient="index")
        st.bar_chart(table[["delta_mse"]])
        st.dataframe(table)
        st.caption(
            "Change in test MSE when one encoded branch is zeroed. All comparisons use the same samples; no retraining."
        )
        st.subheader("Head intervention")
        heads = st.multiselect("Heads to ablate in selected layer", range(cfg["model"]["n_heads"]))
        if st.button("Evaluate head ablation"):
            baseline = (predictions - target).square().mean().item()
            loss = ablate_heads(model, {**inputs, "target": target}, layer, heads)
            st.write(
                f"First test batch: baseline MSE {baseline:.5f}; ablated MSE {loss:.5f}; change {loss - baseline:+.5f}."
            )


if __name__ == "__main__":
    main()
