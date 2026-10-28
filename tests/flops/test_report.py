"""Tests for pure arithmetic report construction."""

import pytest

from torch_harness.flops import summarize_flops


def test_report_is_sorted_and_separates_backward() -> None:
    """Preserve operator totals without conflating backward with other work."""
    counts = {"z": 12, "a": 8}
    report = summarize_flops(counts, backward_flops=12)
    counts["z"] = 0
    assert report.batch_flops == 20
    assert report.non_backward_flops == 8
    assert report.operator_flops == (("a", 8), ("z", 12))


@pytest.mark.parametrize(("count", "backward"), [(-1, 0), (1, -1), (1, 2)])
def test_invalid_counts(count: int, backward: int) -> None:
    """Reject inconsistent arithmetic rather than emitting misleading metrics."""
    with pytest.raises(ValueError, match="FLOPs"):
        summarize_flops({"mm": count}, backward_flops=backward)


def test_empty_report() -> None:
    """Allow zero supported arithmetic without claiming full coverage."""
    assert summarize_flops({}).batch_flops == 0
