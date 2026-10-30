"""Softmax normalization that preserves gradients at high confidence."""

import torch
from torch import Tensor


def _working_logits(logits: Tensor, dim: int) -> Tensor:
    """Validate the class axis and promote low-precision arithmetic."""
    if logits.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise TypeError("logits must have a real floating-point dtype")
    if logits.ndim == 0 or not -logits.ndim <= dim < logits.ndim:
        raise ValueError("dim must select an existing class axis")
    if logits.shape[dim] == 0:
        raise ValueError("the class axis must be nonempty")
    return logits.double() if logits.dtype == torch.float64 else logits.float()


def stable_log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Normalize logits without rounding away the maximum's log-probability.

    Uses the excluded-maximum sum and ``log1p`` from Barron et al. (2026).
    Inputs must be finite, dense, real floating-point tensors. FP16/BF16 inputs
    produce FP32 outputs; FP32/FP64 retain their dtype. Inputs are not mutated.
    ``dim`` is the class axis, including for unbatched inputs.

    Raises:
        TypeError: If the input dtype is unsupported.
        ValueError: If the class axis is missing or empty.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    working = _working_logits(logits, dim)
    maximum_index = working.argmax(dim=dim, keepdim=True)
    maximum = working.gather(dim, maximum_index)
    # A constant at the selected maximum routes its gradient through the others.
    shifted = (working - maximum).scatter(dim, maximum_index, 0.0)
    others = shifted.exp().scatter(dim, maximum_index, 0.0)
    return shifted - others.sum(dim=dim, keepdim=True).log1p()


def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Return probabilities with collapse-resistant backward gradients.

    Shares the input, dtype, error, and axis contract of ``stable_log_softmax``.
    Probabilities near one can still round to one; their gradients are retained
    through the log-probability computation. Inputs are not mutated.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    return stable_log_softmax(logits, dim).exp()
