"""Pure transformations of PyTorch's global operator FLOP counts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class UnmeasuredFlops:
    """Explain why a completed batch report is not available yet."""

    reason: Literal["no_complete_batch"] = "no_complete_batch"


@dataclass(frozen=True)
class FlopReport:
    """Immutable per-rank arithmetic counts for one measured training batch.

    Backward includes checkpoint recomputation. Non-backward includes all other
    counted work between batch hooks, including optimizer work when supported.
    Operator rows contain only PyTorch's Global counts, avoiding module overlap.
    """

    batch_flops: int
    backward_flops: int
    operator_flops: tuple[tuple[str, int], ...]

    @property
    def non_backward_flops(self) -> int:
        """Return arithmetic outside the observed backward regions."""
        return self.batch_flops - self.backward_flops


def summarize_flops(
    counts: Mapping[str, int], *, backward_flops: int = 0
) -> FlopReport:
    """Validate global operator counts and produce a deterministic report.

    Raises:
        ValueError: If counts are negative or backward exceeds the batch total.
    """
    if any(value < 0 for value in counts.values()):
        raise ValueError("Operator FLOPs must be nonnegative")
    total = sum(counts.values())
    if not 0 <= backward_flops <= total:
        raise ValueError("Backward FLOPs must be between zero and batch FLOPs")
    return FlopReport(total, backward_flops, tuple(sorted(counts.items())))
