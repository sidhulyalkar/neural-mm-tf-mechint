"""Streamlit dashboard for Neural MechInt research and deployment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Neural MechInt", page_icon="🧠", layout="wide")
st.title("🧠 Neural MechInt")
st.caption("Observe → Understand → Control → Learn")
st.markdown("""
Mechanistic interpretability becomes operationally valuable when causal evidence can
**predict a failure, control it selectively, and teach the next model version**.
This dashboard only visualizes supplied experiment/receipt artifacts. Missing results remain missing.
""")

with st.sidebar:
    st.header("Artifacts")
    baseline_upload = st.file_uploader("Belief/compliance baseline", type=["json"], key="baseline")
    receipt_upload = st.file_uploader("MechGuard state receipt", type=["json"], key="receipt")
    st.divider()
    st.markdown("**Evidence ladder**")
    st.markdown("Behavior → Decode → Localize → Intervene → Compress → Replicate → Deploy → Learn")

research_tab, state_tab, guard_tab, learn_tab, eval_tab = st.tabs(["Research", "MechState", "MechGuard", "MechTune", "MechEval"])

rows = None
if baseline_upload is not None:
    rows = json.load(baseline_upload)
elif Path("artifacts/belief_compliance_baseline.json").exists():
    rows = json.loads(Path("artifacts/belief_compliance_baseline.json").read_text())
receipt = json.load(receipt_upload) if receipt_upload is not None else None

with research_tab:
    st.subheader("Belief vs. compliance")
    st.write("Does contradictory social pressure override factual evidence, and where does that computation happen?")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        long = df.melt(id_vars=["question"], value_vars=["neutral_logit_diff", "pressured_logit_diff"], var_name="condition", value_name="correct_minus_incorrect_logit")
        st.plotly_chart(px.bar(long, x="question", y="correct_minus_incorrect_logit", color="condition", barmode="group"), use_container_width=True)
    else:
        st.info("No baseline artifact loaded. Run `mechint belief-compliance`. No synthetic findings are substituted.")

with state_tab:
    st.subheader("Internal state trajectory")
    if receipt:
        records = []
        for state in receipt.get("trajectory", {}).get("states", []):
            for name, signal in state.get("signals", {}).items():
                records.append({"layer": state["layer"], "signal": name, "value": signal["value"], "confidence": signal.get("confidence", 1.0)})
        if records:
            state_df = pd.DataFrame(records)
            st.plotly_chart(px.line(state_df, x="layer", y="value", color="signal", markers=True, title="Mechanistic state across depth"), use_container_width=True)
            st.dataframe(state_df, use_container_width=True, hide_index=True)
        else:
            st.warning("The receipt contains no state signals.")
    else:
        st.info("Upload a MechGuard receipt to inspect a real state trajectory.")

with guard_tab:
    st.subheader("Runtime control decision")
    if receipt:
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", receipt.get("decision", "unknown"))
        c2.metric("Risk score", f"{receipt.get('metrics', {}).get('risk_score', 0.0):.3f}")
        c3.metric("Events", int(receipt.get("metrics", {}).get("event_count", 0)))
        st.write("**Reasons**")
        for reason in receipt.get("reasons", []):
            st.write("•", reason)
        st.write("**Intervention**")
        st.json(receipt.get("intervention"))
    else:
        st.info("Receipts make every intervention auditable: state → monitor → policy → action.")

with learn_tab:
    st.subheader("Causal feedback learning")
    st.markdown("Only cases where an intervention **actually improves the target metric** are candidates for training. The next adaptation can then use counterfactual data, representation objectives, Circuit-LoRA, or intervention distillation.")
    st.code("intervention → measured gain → training candidate → targeted adapter → full re-evaluation")

with eval_tab:
    st.subheader("Deployment scorecard")
    st.markdown("A successful controller needs more than accuracy. Track false compliance, mechanistic failure recall, intervention success, collateral KL, intervention rate, and latency overhead.")
    st.dataframe(pd.DataFrame([
        ("Behavior", "task accuracy / false-compliance rate"),
        ("State", "failure recall / calibration / crossover depth"),
        ("Causal", "necessity / sufficiency / collateral KL"),
        ("Operational", "intervention rate / latency / escalation"),
    ], columns=["dimension", "examples"]), use_container_width=True, hide_index=True)
