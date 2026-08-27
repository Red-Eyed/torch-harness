"""BatchNorm layers that reject non-finite inputs before updating state."""

from torch import Tensor, nn
from typing_extensions import override


def _raise_if_non_finite(input: Tensor) -> None:
    """Raise before BatchNorm can consume a non-finite activation."""
    if not bool(input.isfinite().all()):
        raise FloatingPointError("SafeBatchNorm received a non-finite input")


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that rejects non-finite inputs before updating state."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)


class SafeBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that rejects non-finite inputs before updating state."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)


class SafeBatchNorm3d(nn.BatchNorm3d):
    """BatchNorm3d that rejects non-finite inputs before updating state."""

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Normalize a finite input or raise before changing running state."""
        _raise_if_non_finite(input)
        return super().forward(input)
