"""Task-agnostic loss transformations."""

from torch_harness.losses.super_loss import SuperLoss
from torch_harness.losses.uncertainty_weighting import UncertaintyWeighting

__all__ = ["SuperLoss", "UncertaintyWeighting"]
