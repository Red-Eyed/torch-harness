"""PyTorch functional normalization signatures over stable arithmetic."""

import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function_unary

from torch_harness.losses.stable_cross_entropy.normalization import stable_log_softmax


def _normalize(
    input: Tensor,
    dim: int | None,
    stacklevel: int,
    dtype: torch.dtype | None,
    operation: Literal["softmax", "log_softmax"],
) -> Tensor:
    """Match native axis selection and output dtype around the numerical core."""
    if dim is None:
        warnings.warn(
            f"Implicit dimension choice for {operation} has been deprecated. "
            "Change the call to include dim=X as an argument.",
            stacklevel=stacklevel + 1,
        )
        dim = 0 if input.ndim in (0, 1, 3) else 1
    converted = input if dtype is None else input.to(dtype=dtype)
    native = F.softmax if operation == "softmax" else F.log_softmax
    if converted.ndim == 0 or converted.numel() == 0:
        return native(converted, dim=dim)
    if not -converted.ndim <= dim < converted.ndim:
        return native(converted, dim=dim)
    if converted.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return native(converted, dim=dim)
    result = stable_log_softmax(converted, dim)
    if operation == "softmax":
        result = result.exp()
    return result.to(dtype=converted.dtype)


def softmax(
    input: Tensor,
    dim: int | None = None,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Apply stable softmax with PyTorch's signature, axes, and output dtype.

    ``dtype`` casts the input before normalization. Without it, output retains
    the input dtype, including under autocast. Low-precision arithmetic uses
    FP32 internally. Dense finite inputs, scalar tensors, and empty tensors are
    supported; omitted dimensions emit PyTorch's deprecation warning. Invalid
    axes and dtypes raise native PyTorch errors. Inputs are not mutated.
    Casting back to low precision can underflow small gradients.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    return _normalize(input, dim, _stacklevel, dtype, "softmax")


def log_softmax(
    input: Tensor,
    dim: int | None = None,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Apply stable log-softmax with PyTorch's signature and dtype behavior.

    Shares ``softmax``'s dense finite input, dtype, autocast, error, and mutation
    contracts. Excluding the selected maximum preserves confident derivatives;
    the result is cast to the requested or input dtype. Scalar and empty inputs
    follow native behavior. Omitted dimensions emit a deprecation warning.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            log_softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype
        )
    return _normalize(input, dim, _stacklevel, dtype, "log_softmax")
