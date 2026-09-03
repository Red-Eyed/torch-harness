"""Task-agnostic SuperLoss for robust curriculum learning."""

from math import e, isfinite
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn
from typing_extensions import override

_Reduction = Literal["none", "mean", "sum"]
_VALID_REDUCTIONS: tuple[_Reduction, ...] = ("none", "mean", "sum")
_LAMBERT_W_ITERATIONS = 12


def _principal_lambert_w(input: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Evaluate the real principal Lambert W branch on its supported domain."""
    working_input = input if input.dtype == torch.float64 else input.float()
    branch_point = working_input.new_tensor(-1.0 / e)
    at_branch_point = working_input <= branch_point
    iteration_input = torch.where(
        at_branch_point,
        torch.zeros_like(working_input),
        working_input,
    )

    negative_initial = -1.0 + torch.sqrt(
        torch.clamp_min(2.0 * (1.0 + e * iteration_input), 0.0)
    )
    positive_initial = torch.log1p(torch.clamp_min(iteration_input, 0.0))
    large_input = torch.clamp_min(iteration_input, 3.0)
    large_initial = torch.log(large_input) - torch.log(torch.log(large_input))
    estimate = torch.where(
        iteration_input < 0.0,
        negative_initial,
        torch.where(iteration_input > 3.0, large_initial, positive_initial),
    )

    for _ in range(_LAMBERT_W_ITERATIONS):
        exponential = estimate.exp()
        residual = estimate * exponential - iteration_input
        denominator = exponential * (estimate + 1.0) - (
            (estimate + 2.0) * residual / (2.0 * estimate + 2.0)
        )
        estimate = estimate - residual / denominator

    result = torch.where(at_branch_point, -torch.ones_like(estimate), estimate)
    return result.to(dtype=input.dtype)


def _optimal_log_confidence(
    loss: Float[Tensor, "..."],
    threshold: float,
    regularization: float,
) -> Float[Tensor, "..."]:
    """Compute the detached log-confidence that minimizes SuperLoss."""
    working_loss = loss if loss.dtype == torch.float64 else loss.float()
    normalized_loss = (working_loss - threshold) / regularization
    minimum = normalized_loss.new_tensor(-2.0 / e)
    lambert_input = 0.5 * torch.clamp_min(normalized_loss, minimum)
    log_confidence = -_principal_lambert_w(lambert_input)
    return log_confidence.to(dtype=loss.dtype)


def _reduce(
    loss: Float[Tensor, "..."],
    reduction: _Reduction,
) -> Float[Tensor, "..."]:
    """Apply a validated PyTorch-style reduction to the transformed loss."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class SuperLoss(nn.Module):
    """Turn per-item task losses into robust curriculum-weighted losses.

    Implements "SuperLoss: A Generic Loss for Robust Curriculum Learning"
    (Castells, Weinzaepfel, and Revaud, NeurIPS 2020).

    The input must contain unreduced floating-point task losses. During
    backpropagation, the optimal confidence is treated as a constant, so each
    task-loss gradient is scaled by its confidence.

    Args:
        threshold: Boundary between easy and hard samples in task-loss units.
        regularization: Positive strength controlling confidence variation.
        reduction: Reduction applied after transforming each loss.

    Raises:
        ValueError: If a numeric parameter is invalid or the reduction is unknown.
    """

    def __init__(
        self,
        threshold: float,
        regularization: float = 1.0,
        reduction: _Reduction = "mean",
    ) -> None:
        """Initialize SuperLoss with explicit, task-scale-aware parameters."""
        super().__init__()
        if not isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not isfinite(regularization) or regularization <= 0.0:
            raise ValueError("regularization must be finite and greater than zero")
        if reduction not in _VALID_REDUCTIONS:
            raise ValueError(f"unsupported reduction: {reduction!r}")

        self.threshold = threshold
        self.regularization = regularization
        self.reduction = reduction

    @override
    def forward(self, loss: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Transform unreduced task losses and apply the configured reduction.

        Args:
            loss: Floating-point task losses with any shape.

        Returns:
            A tensor with the input shape for ``none`` reduction, otherwise a
            scalar tensor.

        Raises:
            TypeError: If ``loss`` is not a floating-point tensor.
        """
        if not loss.is_floating_point():
            raise TypeError("SuperLoss requires floating-point task losses")

        with torch.no_grad():
            log_confidence = _optimal_log_confidence(
                loss,
                threshold=self.threshold,
                regularization=self.regularization,
            )
            confidence = log_confidence.exp()

        transformed = confidence * (loss - self.threshold)
        transformed = transformed + self.regularization * log_confidence.square()
        return _reduce(transformed, self.reduction)
