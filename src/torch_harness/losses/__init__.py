"""Task-agnostic loss transformations."""

from torch_harness.losses.adaptive_balancing import (
    BalancerConfig,
    LossBalancer,
    LossTarget,
    weighted_loss,
)
from torch_harness.losses.stable_cross_entropy import (
    StableCrossEntropyLoss,
    stable_cross_entropy,
    stable_distillation_cross_entropy,
    stable_log_softmax,
    stable_softmax,
)
from torch_harness.losses.super_loss import SuperLoss
from torch_harness.losses.uncertainty_weighting import UncertaintyWeighting

__all__ = [
    "BalancerConfig",
    "LossBalancer",
    "LossTarget",
    "StableCrossEntropyLoss",
    "SuperLoss",
    "UncertaintyWeighting",
    "stable_cross_entropy",
    "stable_distillation_cross_entropy",
    "stable_log_softmax",
    "stable_softmax",
    "weighted_loss",
]
