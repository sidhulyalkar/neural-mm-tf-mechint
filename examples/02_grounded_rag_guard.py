"""Reference wiring for the MechGuard grounded-RAG profile.

This example expects calibrated direction tensors produced by your own held-out
activation dataset. It intentionally does not ship fake directions.
"""
from pathlib import Path

import torch

from neural_mechint.adapters import HFCausalLMAdapter
from neural_mechint.applications.grounded_rag import DEMO_SCENARIOS, build_grounded_rag_guard

MODEL = "google/gemma-3-270m-it"
DIRECTIONS = Path("artifacts/directions")

adapter = HFCausalLMAdapter.from_pretrained(MODEL)
guard = build_grounded_rag_guard(
    adapter,
    retrieval_direction=torch.load(DIRECTIONS / "retrieval.pt", map_location="cpu", weights_only=True),
    pressure_direction=torch.load(DIRECTIONS / "pressure.pt", map_location="cpu", weights_only=True),
    commitment_direction=torch.load(DIRECTIONS / "commitment.pt", map_location="cpu", weights_only=True),
    correction_direction=torch.load(DIRECTIONS / "correction.pt", map_location="cpu", weights_only=True),
    intervention_layer=max(0, adapter.n_layers - 4),
    monitor_layers=list(range(max(0, adapter.n_layers - 8), adapter.n_layers)),
)

scenario = DEMO_SCENARIOS[0]
result = guard.generate(scenario.prompt(), prompt_id=scenario.scenario_id, do_sample=False)
print("decision:", result.decision.action)
print("response:", result.response)
print("receipt:", result.receipt.to_dict())
