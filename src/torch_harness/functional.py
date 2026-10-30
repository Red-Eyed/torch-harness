"""PyTorch functional replacements using numerically stable normalization."""

from torch_harness.losses.stable_cross_entropy._functional_cross_entropy import (
    cross_entropy,
)
from torch_harness.losses.stable_cross_entropy._functional_normalization import (
    log_softmax,
    softmax,
)

__all__ = ["cross_entropy", "log_softmax", "softmax"]
