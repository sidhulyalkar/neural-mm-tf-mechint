"""Streamlit research dashboard for Neural MechInt Lab artifacts.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Neural MechInt Lab", page_icon="🧠", layout="wide")

st.title("🧠 Neural MechInt Lab")
st.caption("Mechanistic interpretability from decodability to causal circuit evidence")

st.markdown(
    """
**Claim discipline:** attention maps and probes nominate hypotheses. Activation patching,
ablation, steering, and circuit-retention tests decide whether those hypotheses deserve
causal language.
"""
)

with st.sidebar:
    st.header("Evidence ladder")
    st.markdown(
        """
1. **Behavior** — define a scalar target
2. **Decode** — test information presence
3. **Localize** — nominate layers/features
4. **Intervene** — patch, ablate, steer
5. **Compress** — necessity + sufficiency
6. **Replicate** — prompts, seeds, models
"""
    )
    artifact = st.file_uploader("Load baseline JSON", type=["json"])

rows = None
if artifact is not None:
    rows = json.load(artifact)
else:
    default = Path("artifacts/belief_compliance_baseline.json")
    if default.exists():
        rows = json.loads(default.read_text())

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Flagship study: belief vs. compliance")
    st.markdown(
        """
A factual question is asked twice: once neutrally and once after the user confidently
asserts an incorrect answer and asks the model to agree. The central metric is the
**correct-minus-incorrect next-token logit difference**.
"""
    )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        long = df.melt(
            id_vars=["question"],
            value_vars=["neutral_logit_diff", "pressured_logit_diff"],
            var_name="condition",
            value_name="correct_minus_incorrect_logit",
        )
        fig = px.bar(
            long,
            x="question",
            y="correct_minus_incorrect_logit",
            color="condition",
            barmode="group",
            title="Behavioral pressure effect before any mechanistic interpretation",
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Δ logit", legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No artifact loaded. Run `mechint belief-compliance` or upload its JSON output. "
            "The dashboard intentionally does not display fabricated demo findings."
        )

with right:
    st.subheader("What would count as a mechanism?")
    st.markdown(
        """
- **Probe:** pressure is decodable from a held-out activation.
- **Patch:** neutral residual state restores the factual answer under pressure.
- **Steer:** positive/negative pressure directions yield a coherent dose-response.
- **Specificity:** the effect beats equal-norm random directions and preserves unrelated behavior.
- **Circuit:** a sparse feature/component subset is necessary and approximately sufficient.
- **Replication:** the effect survives unseen facts, paraphrases, seeds, and model sizes.
"""
    )

    st.subheader("Frontier bridge")
    st.markdown(
        """
The research roadmap connects dense residual interventions to Gemma Scope 2 sparse
features/transcoders, attribution graphs, cross-model diffing, and MIB-style circuit
faithfulness evaluation.
"""
    )

st.divider()
st.subheader("Research sequence")
roadmap = pd.DataFrame(
    [
        (1, "Behavior atlas", "Does pressure actually move factual logits?"),
        (2, "Layer localization", "Where is pressure/fact information decodable?"),
        (3, "Activation patching", "Which states causally restore the clean behavior?"),
        (4, "Steering", "Can the representation control behavior bidirectionally?"),
        (5, "Sparse mediation", "Can a small SAE/transcoder feature set explain the effect?"),
        (6, "Attribution graph", "How does evidence flow into answer selection?"),
        (7, "Model diff", "What changes after instruction tuning or scaling?"),
    ],
    columns=["stage", "experiment", "scientific question"],
)
st.dataframe(roadmap, use_container_width=True, hide_index=True)
