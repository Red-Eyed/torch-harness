"""Hard-target cross-entropy built on collapse-resistant normalization."""

from typing import Literal

import torch
from torch import Tensor

from torch_harness.losses.stable_cross_entropy.normalization import stable_log_softmax

Reduction = Literal["none", "mean", "sum"]


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    """Apply a validated reduction without changing per-item semantics."""
    match reduction:
        case "none":
            return loss
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case _:
            raise ValueError(f"unsupported reduction: {reduction!r}")


def stable_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    dim: int = -1,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute hard-label cross-entropy while retaining confident gradients.

    ``targets`` must be int64 class indices on the logits' device, with the
    class axis removed from their shape. Indices must lie in the class range.
    Uses ``stable_log_softmax``'s finite-input and accumulation dtype contract.
    No class weighting, ignored labels, or label smoothing is applied. Inputs
    are not mutated. Empty mean reductions return NaN, as in PyTorch.

    Raises:
        TypeError: If logits or target dtypes are unsupported.
        ValueError: If shapes, devices, the axis, or reduction are invalid.
        RuntimeError: If a target index lies outside the class range.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    log_probabilities = stable_log_softmax(logits, dim)
    axis = dim % logits.ndim
    expected_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
    if targets.dtype != torch.int64:
        raise TypeError("targets must contain int64 class indices")
    if targets.shape != expected_shape:
        raise ValueError("targets must have the logits shape without the class axis")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    losses = -log_probabilities.gather(axis, targets.unsqueeze(axis)).squeeze(axis)
    return _reduce(losses, reduction)
