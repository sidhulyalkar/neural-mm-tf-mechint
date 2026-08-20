"""Minimal example: cache activations, localize causally, then derive a steering direction."""

from neural_mechint.adapters import HFCausalLMAdapter
from neural_mechint.experiments import BeliefComplianceExperiment

adapter = HFCausalLMAdapter.from_pretrained("google/gemma-3-270m-it")
experiment = BeliefComplianceExperiment(adapter)

eligible = experiment.eligible_facts()
if not eligible:
    raise RuntimeError("No single-token fact contrasts were available for this tokenizer")

fact = eligible[0]
print("Fact:", fact.fact.question)
print("Patch scan:")
for row in experiment.patch_fact(fact):
    print(row)

middle_layer = adapter.n_layers // 2
print("Probe:", experiment.probe_layer(middle_layer))
print("Pressure direction shape:", tuple(experiment.pressure_direction(middle_layer).shape))
