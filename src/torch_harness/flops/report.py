"""Pure transformations of PyTorch's global forward operator FLOP counts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class UnmeasuredFlops:
    """Explain why a completed forward report is not available yet."""

    reason: Literal["no_complete_forward"] = "no_complete_forward"


@dataclass(frozen=True)
class FlopReport:
    """Immutable theoretical FLOPs for one observed model forward on one rank.

    Input shapes follow positional then keyword tensor encounter order. Batch
    size follows the first non-scalar tensor's leading dimension, or one when
    none exists. This convention must be checked for non-batch-first models.
    Counts describe the whole observed input, not automatically one sample.
    """

    forward_flops: int
    input_shapes: tuple[tuple[int, ...], ...]
    batch_size: int
    operator_flops: tuple[tuple[str, int], ...]
    multiply_add_flops: Literal[2] = 2


def summarize_flops(
    counts: Mapping[str, int], *, input_shapes: tuple[tuple[int, ...], ...] = ()
) -> FlopReport:
    """Validate global counts and build a deterministic forward report.

    Module totals must not be summed into counts because their attribution
    overlaps. Input shapes are metadata only; no tensors are retained.

    Raises:
        ValueError: If an operator count or input dimension is negative.
    """
    if any(value < 0 for value in counts.values()):
        raise ValueError("Operator FLOPs must be nonnegative")
    if any(dimension < 0 for shape in input_shapes for dimension in shape):
        raise ValueError("Input dimensions must be nonnegative")
    batch_size = next((shape[0] for shape in input_shapes if shape), 1)
    return FlopReport(
        sum(counts.values()), input_shapes, batch_size, tuple(sorted(counts.items()))
    )
