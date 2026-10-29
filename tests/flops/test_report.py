"""Tests for pure forward report construction."""

import pytest

from torch_harness.flops import summarize_flops


def test_report_is_sorted_and_retains_input_metadata() -> None:
    """Preserve global operator totals and the observed batch convention."""
    counts = {"z": 12, "a": 8}
    report = summarize_flops(counts, input_shapes=((2, 4), (2,)))
    counts["z"] = 0
    assert report.forward_flops == 20
    assert report.input_shapes == ((2, 4), (2,))
    assert report.batch_size == 2
    assert report.multiply_add_flops == 2
    assert report.operator_flops == (("a", 8), ("z", 12))


def test_invalid_counts() -> None:
    """Reject negative arithmetic rather than emitting misleading metrics."""
    with pytest.raises(ValueError, match="FLOPs"):
        summarize_flops({"mm": -1})


def test_invalid_input_dimensions() -> None:
    """Reject dimensions that cannot describe observed inputs."""
    with pytest.raises(ValueError, match="dimensions"):
        summarize_flops({}, input_shapes=((-1, 4),))


@pytest.mark.parametrize("shapes", [(), ((),)])
def test_no_batched_input(shapes: tuple[tuple[int, ...], ...]) -> None:
    """Use a single invocation convention for no tensor inputs or scalar inputs."""
    report = summarize_flops({}, input_shapes=shapes)
    assert report.forward_flops == 0
    assert report.batch_size == 1
