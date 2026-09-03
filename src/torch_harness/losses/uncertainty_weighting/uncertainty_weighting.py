"""Learn task-level uncertainty weights for multi-task losses."""

from math import expm1, isfinite, log

import torch
import torch.nn.functional as functional
from jaxtyping import Float
from torch import Tensor, nn
from typing_extensions import override

_DEFAULT_EPSILON = 1e-6


def _inverse_softplus(value: float) -> float:
    """Return the numerically stable inverse softplus of a positive scalar."""
    return value + log(-expm1(-value))


def _calculation_dtype(*tensors: Tensor) -> torch.dtype:
    """Choose at least float32 while preserving an explicitly double module or input."""
    if any(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def _variances(
    raw_uncertainties: Float[Tensor, "n"],
    epsilon: float,
    dtype: torch.dtype,
) -> Float[Tensor, "n"]:
    """Map unconstrained task parameters to strictly positive variances."""
    return functional.softplus(raw_uncertainties.to(dtype=dtype)) + epsilon


class UncertaintyWeighting(nn.Module):
    """Combine peer task losses using learned homoscedastic uncertainties.

    For each task, the module learns an unconstrained parameter ``r`` and maps it
    to a positive variance ``v = softplus(r) + epsilon``. The combined loss is
    ``sum(loss / v + log(v))`` and the effective task weight is ``1 / v``.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics" by Kendall, Gal, and Cipolla (CVPR 2018):
    https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html

    Args:
        num_tasks: Number of scalar losses expected by :meth:`forward`.
        epsilon: Positive variance floor. It must be smaller than one so every
            effective weight can be initialized to one.

    Raises:
        TypeError: If ``num_tasks`` is not an integer.
        ValueError: If ``num_tasks`` or ``epsilon`` is outside its valid range.
    """

    def __init__(self, num_tasks: int, epsilon: float = _DEFAULT_EPSILON) -> None:
        """Initialize every task with unit effective weight."""
        super().__init__()
        if isinstance(num_tasks, bool) or not isinstance(num_tasks, int):
            raise TypeError("num_tasks must be an integer")
        if num_tasks <= 0:
            raise ValueError("num_tasks must be greater than zero")
        if not isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be finite and between zero and one")

        initial_raw_uncertainty = _inverse_softplus(1.0 - epsilon)
        self.num_tasks = num_tasks
        self.epsilon = epsilon
        self.raw_uncertainties = nn.Parameter(
            torch.full((num_tasks,), initial_raw_uncertainty)
        )

    @property
    def effective_weights(self) -> Float[Tensor, "n"]:
        """Return detached inverse variances for monitoring task contributions."""
        dtype = _calculation_dtype(self.raw_uncertainties)
        with torch.autocast(
            device_type=self.raw_uncertainties.device.type,
            enabled=False,
        ):
            variances = _variances(
                self.raw_uncertainties,
                epsilon=self.epsilon,
                dtype=dtype,
            )
            return variances.reciprocal().detach()

    @override
    def forward(self, losses: Float[Tensor, "n"]) -> Float[Tensor, ""]:
        """Return the scalar uncertainty-weighted sum of task losses.

        Numerically sensitive operations run outside autocast in at least
        float32. Float64 inputs or module parameters preserve double precision.

        Args:
            losses: One-dimensional floating-point tensor with one scalar loss
                per configured task.

        Returns:
            Scalar loss with gradients to both ``losses`` and the learned
            uncertainty parameters.

        Raises:
            TypeError: If ``losses`` is not a floating-point tensor.
            ValueError: If ``losses`` is not rank one or has the wrong length.
        """
        if not isinstance(losses, Tensor):
            raise TypeError("losses must be a torch.Tensor")
        if losses.ndim != 1:
            raise ValueError(f"losses must have rank 1; got rank {losses.ndim}")
        if losses.shape[0] != self.num_tasks:
            raise ValueError(
                f"losses must contain {self.num_tasks} tasks; got {losses.shape[0]}"
            )
        if not losses.is_floating_point():
            raise TypeError("losses must be a floating-point tensor")

        dtype = _calculation_dtype(losses, self.raw_uncertainties)
        with torch.autocast(device_type=losses.device.type, enabled=False):
            working_losses = losses.to(dtype=dtype)
            variances = _variances(
                self.raw_uncertainties,
                epsilon=self.epsilon,
                dtype=dtype,
            )
            return (working_losses / variances + variances.log()).sum()
