"""Native CrossEntropyLoss lifecycle with numerically stable forward behavior."""

from torch import Tensor, nn
from typing_extensions import override

from torch_harness.losses.stable_cross_entropy._functional_cross_entropy import (
    cross_entropy,
)


class StableCrossEntropyLoss(nn.CrossEntropyLoss):
    """Replace CrossEntropyLoss while preserving its constructor and state layout.

    Inherits native class-weight buffer registration, device/dtype moves,
    reduction configuration, ignored labels, and label smoothing. Forward uses
    stable normalization with the compatible functional cross-entropy contract.
    Existing CrossEntropyLoss state dictionaries load without migration. Inputs
    are not mutated; invalid inputs raise the functional API's errors.

    References:
        Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
        "A Fix for Softmax and Cross-Entropy" (2026), Section 2.
        https://arxiv.org/abs/2610.31642
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Apply stable cross-entropy with the inherited native configuration."""
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
