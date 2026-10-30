"""PyTorch-compatible weighted and smoothed stable cross-entropy."""

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function_variadic

from torch_harness.losses.stable_cross_entropy._functional_normalization import (
    log_softmax,
)
from torch_harness.losses.stable_cross_entropy.cross_entropy import Reduction, _reduce


def _resolve_reduction(
    size_average: bool | None, reduce: bool | None, reduction: str
) -> Reduction:
    """Honor deprecated flags with PyTorch's precedence and warning."""
    if size_average is not None or reduce is not None:
        average = True if size_average is None else size_average
        reducing = True if reduce is None else reduce
        reduction = ("mean" if average else "sum") if reducing else "none"
        warnings.warn(
            "size_average and reduce args will be deprecated, "
            f"please use reduction='{reduction}' instead.",
            stacklevel=3,
        )
    match reduction:
        case "none" | "mean" | "sum":
            return reduction
        case _:
            raise ValueError(f"{reduction} is not a valid value for reduction")


def _autocast_enabled(input: Tensor) -> bool:
    """Query device autocast while retaining support for older PyTorch versions."""
    try:
        return torch.is_autocast_enabled(input.device.type)
    except TypeError:
        if input.device.type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()


def _class_weights(weight: Tensor, input: Tensor, dim: int) -> Tensor:
    """Broadcast a class vector over native unbatched or spatial layouts."""
    shape = [1] * input.ndim
    shape[dim] = input.shape[dim]
    return weight.reshape(shape)


def _probability_loss(
    log_probabilities: Tensor,
    target: Tensor,
    weight: Tensor | None,
    dim: int,
    reduction: Reduction,
    label_smoothing: float,
) -> Tensor:
    """Apply native soft-target weighting and per-item mean normalization."""
    if not target.is_floating_point():
        raise RuntimeError(
            "Expected floating point type for target with class probabilities"
        )
    if weight is not None and (
        weight.ndim != 1 or weight.numel() != log_probabilities.shape[dim]
    ):
        raise RuntimeError("weight tensor must contain one weight per class")
    probabilities = target
    if label_smoothing > 0:
        probabilities = (
            target * (1.0 - label_smoothing) + label_smoothing / target.shape[dim]
        )
    weighted = log_probabilities
    if weight is not None:
        weighted = weighted * _class_weights(weight, weighted, dim)
    return _reduce(-(probabilities * weighted).sum(dim=dim), reduction)


def _index_loss(
    log_probabilities: Tensor,
    target: Tensor,
    weight: Tensor | None,
    dim: int,
    ignore_index: int,
    reduction: Reduction,
    label_smoothing: float,
) -> Tensor:
    """Combine native NLL semantics with an optional weighted uniform target."""
    nll = F.nll_loss(
        log_probabilities,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if label_smoothing == 0:
        return nll
    weighted = log_probabilities
    if weight is not None:
        weighted = weighted * _class_weights(weight, weighted, dim)
    ignored = target == ignore_index
    smooth = -weighted.sum(dim=dim).masked_fill(ignored, 0.0)
    if reduction == "mean":
        if weight is None:
            denominator = (~ignored).sum()
        else:
            indices = target.long().masked_fill(ignored, 0)
            denominator = weight[indices].masked_fill(ignored, 0.0).sum()
        smooth = smooth.sum() / denominator
    else:
        smooth = _reduce(smooth, reduction)
    return nll * (1.0 - label_smoothing) + smooth * (
        label_smoothing / log_probabilities.shape[dim]
    )


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute stable cross-entropy with PyTorch's functional call contract.

    Accepts ``(C)``, ``(N, C)``, or ``(N, C, ...)`` dense finite logits. Targets
    are class indices with the class axis removed, or floating probabilities
    with the input shape. Supports class weights, ignored indices, smoothing,
    all native reductions, and deprecated reduction flags. Weighted index means
    divide by non-ignored target weights; probability means divide by item count.
    Probability targets are not validated as distributions, matching PyTorch.

    Index losses retain the logits dtype outside autocast and use FP32 for
    low-precision logits under CPU/CUDA autocast. Probability targets and weights
    follow native dtype promotion. Low-precision normalization uses FP32
    internally. Inputs are not mutated. Probability targets remain
    differentiable, but cannot provide the separate logits-based distillation
    cancellation fix. Higher derivatives follow ordinary autograd.

    Invalid weights, targets, shapes, and smoothing raise PyTorch-style errors;
    an invalid reduction raises ValueError. Non-finite and sparse tensors are
    outside the supported contract. Compiler numerical accuracy is backend-specific.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(
            cross_entropy,
            (input, target, weight),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    resolved = _resolve_reduction(size_average, reduce, reduction)
    if not 0.0 <= label_smoothing <= 1.0:
        raise RuntimeError("label_smoothing must be between 0.0 and 1.0")
    dim = 0 if input.ndim == 1 else 1
    working = input
    if _autocast_enabled(input):
        if input.dtype in (torch.float16, torch.bfloat16):
            working = input.float()
        if target.is_floating_point() and target.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            target = target.float()
        if weight is not None and weight.dtype in (torch.float16, torch.bfloat16):
            weight = weight.float()
    log_probabilities = log_softmax(working, dim=dim)
    if input.shape == target.shape:
        if ignore_index >= 0:
            raise RuntimeError(
                "ignore_index is not supported for floating point target"
            )
        return _probability_loss(
            log_probabilities, target, weight, dim, resolved, label_smoothing
        )
    return _index_loss(
        log_probabilities, target, weight, dim, ignore_index, resolved, label_smoothing
    )
