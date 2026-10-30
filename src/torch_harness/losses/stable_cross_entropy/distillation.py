"""Logit-space distillation with cancellation-resistant student gradients."""

import torch
from torch import Tensor

from torch_harness.losses.stable_cross_entropy.cross_entropy import Reduction, _reduce
from torch_harness.losses.stable_cross_entropy.normalization import (
    _working_logits,
    stable_log_softmax,
)


def _student_gradient(
    student: Tensor, teacher: Tensor, q: Tensor, p: Tensor, dim: int
) -> Tensor:
    """Compute probability differences accurately near equal logits."""
    delta = student - teacher
    centered = delta - (p * delta).sum(dim=dim, keepdim=True)
    near = centered.abs().amax(dim=dim, keepdim=True) < 0.5
    # Bound the unused branch too: where evaluates both sides eagerly.
    safe_centered = torch.where(near, centered, 0.0)
    log_ratio = (
        safe_centered - (p * safe_centered.expm1()).sum(dim=dim, keepdim=True).log1p()
    )
    difference = torch.where(near, p * log_ratio.expm1(), q - p)
    return difference - q * difference.sum(dim=dim, keepdim=True)


def stable_distillation_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    dim: int = -1,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute teacher cross-entropy with a stable first-order student gradient.

    Teacher and student must have identical shapes, dtype, and device. Both
    contain finite logits, with the class axis selected by ``dim``. FP16/BF16
    computation and output use FP32; FP64 is preserved. The forward value is
    cross-entropy, so equal logits yield teacher entropy, not zero loss.

    The teacher is detached. The prescribed student gradient uses logit-space
    differences near equality and a normalized probability difference elsewhere.
    Higher-order derivatives are unsupported: the substituted gradient is
    detached. Apply temperature scaling to logits outside this function.
    Inputs are not mutated.

    Raises:
        TypeError: If logits have unsupported or mismatched dtypes.
        ValueError: If shapes, devices, the axis, or reduction are invalid.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 3.
        https://arxiv.org/abs/2610.31642
    """
    if student_logits.dtype != teacher_logits.dtype:
        raise TypeError("student and teacher must have the same dtype")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher must have the same shape")
    if student_logits.device != teacher_logits.device:
        raise ValueError("student and teacher must be on the same device")
    student = _working_logits(student_logits, dim)
    teacher = _working_logits(teacher_logits.detach(), dim)
    with torch.no_grad():
        log_q = stable_log_softmax(student, dim)
        q = log_q.exp()
        p = stable_log_softmax(teacher, dim).exp()
        gradient = _student_gradient(student, teacher, q, p, dim)
        forward_loss = -(p * log_q).sum(dim=dim)
    # The zero-valued term supplies only the prescribed first-order gradient.
    loss = forward_loss + (gradient * (student - student.detach())).sum(dim=dim)
    return _reduce(loss, reduction)
