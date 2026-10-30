"""Numerically stable normalization and cross-entropy from Barron et al."""

from torch_harness.losses.stable_cross_entropy.cross_entropy import stable_cross_entropy
from torch_harness.losses.stable_cross_entropy.distillation import (
    stable_distillation_cross_entropy,
)
from torch_harness.losses.stable_cross_entropy.loss import StableCrossEntropyLoss
from torch_harness.losses.stable_cross_entropy.normalization import (
    stable_log_softmax,
    stable_softmax,
)

__all__ = [
    "StableCrossEntropyLoss",
    "stable_cross_entropy",
    "stable_distillation_cross_entropy",
    "stable_log_softmax",
    "stable_softmax",
]
