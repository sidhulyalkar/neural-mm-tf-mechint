from __future__ import annotations

import torch
import torch.nn.functional as F


def grounding_preservation_loss(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Keep a trusted-evidence representation stable under irrelevant prompt perturbations."""
    if current.shape != reference.shape:
        raise ValueError("current and reference must have matching shape")
    return 1.0 - F.cosine_similarity(current.float(), reference.float(), dim=-1).mean()


def separation_loss(lhs: torch.Tensor, rhs: torch.Tensor, *, margin: float = 0.25) -> torch.Tensor:
    """Penalize excessive alignment between two representation families."""
    if lhs.shape != rhs.shape:
        raise ValueError("lhs and rhs must have matching shape")
    similarity = F.cosine_similarity(lhs.float(), rhs.float(), dim=-1)
    return torch.relu(similarity - margin).mean()


def intervention_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, *, temperature: float = 2.0) -> torch.Tensor:
    """Distill an intervened teacher distribution into an un-intervened student."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have matching shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_p = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (temperature**2)


def state_matching_loss(student_state: torch.Tensor, teacher_state: torch.Tensor) -> torch.Tensor:
    if student_state.shape != teacher_state.shape:
        raise ValueError("student_state and teacher_state must have matching shape")
    return F.mse_loss(student_state.float(), teacher_state.float())
