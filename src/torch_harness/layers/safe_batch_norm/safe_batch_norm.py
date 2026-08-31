"""BatchNorm layers that reject non-finite inputs before updating state."""

from typing import Self, TypeVar

from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing_extensions import override

_BatchNormT = TypeVar("_BatchNormT", bound=_BatchNorm)


def _raise_if_non_finite(input: Tensor) -> None:
    """Raise before BatchNorm can consume a non-finite activation."""
    if not bool(input.isfinite().all()):
        raise FloatingPointError("SafeBatchNorm received a non-finite input")


def _reference_tensor(batch_norm: _BatchNorm) -> Tensor | None:
    """Find state whose device and dtype characterize a BatchNorm layer."""
    for tensor in (
        batch_norm.weight,
        batch_norm.bias,
        batch_norm.running_mean,
        batch_norm.running_var,
    ):
        if tensor is not None:
            return tensor
    return None


def _copy_requires_grad(source: _BatchNorm, target: _BatchNorm) -> None:
    """Preserve whether each affine parameter participates in optimization."""
    if source.weight is not None and target.weight is not None:
        target.weight.requires_grad_(source.weight.requires_grad)
    if source.bias is not None and target.bias is not None:
        target.bias.requires_grad_(source.bias.requires_grad)


def _convert_batch_norm(
    source: _BatchNorm,
    target_cls: type[_BatchNormT],
) -> _BatchNormT:
    """Rebuild BatchNorm as a safe subclass with equivalent persistent state."""
    reference = _reference_tensor(source)
    device = reference.device if reference is not None else None
    dtype = reference.dtype if reference is not None else None
    target = target_cls(
        num_features=source.num_features,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
        device=device,
        dtype=dtype,
    )

    if source.affine and source.bias is None:
        target.register_parameter("bias", None)

    target.load_state_dict(source.state_dict())
    target.train(source.training)
    _copy_requires_grad(source, target)
    return target


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that rejects non-finite inputs before updating state."""

    @classmethod
    def from_batch_norm(cls, batch_norm: nn.BatchNorm1d) -> Self:
        """Rebuild a BatchNorm1d while preserving configuration and state."""
        return _convert_batch_norm(batch_norm, cls)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)


class SafeBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that rejects non-finite inputs before updating state."""

    @classmethod
    def from_batch_norm(cls, batch_norm: nn.BatchNorm2d) -> Self:
        """Rebuild a BatchNorm2d while preserving configuration and state."""
        return _convert_batch_norm(batch_norm, cls)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)


class SafeBatchNorm3d(nn.BatchNorm3d):
    """BatchNorm3d that rejects non-finite inputs before updating state."""

    @classmethod
    def from_batch_norm(cls, batch_norm: nn.BatchNorm3d) -> Self:
        """Rebuild a BatchNorm3d while preserving configuration and state."""
        return _convert_batch_norm(batch_norm, cls)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)
