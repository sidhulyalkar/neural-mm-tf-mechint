import torch

from neural_mechint.mechtune import LayerEffect, generate_counterfactuals, grounding_preservation_loss, intervention_distillation_loss, plan_circuit_lora, separation_loss


def test_counterfactual_generator_is_factorial():
    examples = generate_counterfactuals([("Capital of France?", "Paris", "Lyon")], pressure_styles=("mild", "authority"))
    assert len(examples) == 2
    assert {example.pressure_style for example in examples} == {"mild", "authority"}


def test_representation_losses_have_expected_ordering():
    x = torch.tensor([[1.0, 0.0]])
    assert grounding_preservation_loss(x, x).item() < 1e-6
    assert separation_loss(x, x, margin=0.25).item() > separation_loss(x, -x, margin=0.25).item()


def test_distillation_zero_for_equal_logits():
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    assert intervention_distillation_loss(logits, logits).item() < 1e-6


def test_circuit_lora_selects_high_effect_low_collateral_layers():
    plan = plan_circuit_lora([LayerEffect(5, 0.2, 0.01), LayerEffect(9, 0.9, 0.02), LayerEffect(12, 1.2, 0.9), LayerEffect(14, -0.8, 0.03)], top_k=2, max_collateral_kl=0.05)
    assert plan.target_layers == (9, 14)
