"""Tests for BatchNorm layers that reject non-finite activations."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from torch_utils import SafeBatchNorm1d, SafeBatchNorm2d, SafeBatchNorm3d

BatchNormType = type[_BatchNorm]

BATCH_NORM_CASES: tuple[tuple[BatchNormType, BatchNormType, tuple[int, ...]], ...] = (
    (nn.BatchNorm1d, SafeBatchNorm1d, (4, 3)),
    (nn.BatchNorm1d, SafeBatchNorm1d, (4, 3, 5)),
    (nn.BatchNorm2d, SafeBatchNorm2d, (4, 3, 5, 7)),
    (nn.BatchNorm3d, SafeBatchNorm3d, (4, 3, 2, 5, 7)),
)


def _assert_running_stats_equal(actual: _BatchNorm, expected: _BatchNorm) -> None:
    """Assert exact equality for all persistent BatchNorm statistics."""
    torch.testing.assert_close(
        actual.running_mean,
        expected.running_mean,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual.running_var, expected.running_var, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.num_batches_tracked,
        expected.num_batches_tracked,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("batch_norm_factory", "safe_batch_norm_factory", "input_shape"),
    BATCH_NORM_CASES,
)
def test_finite_forward_matches_batch_norm(
    batch_norm_factory: BatchNormType,
    safe_batch_norm_factory: BatchNormType,
    input_shape: tuple[int, ...],
) -> None:
    """Verify finite outputs, state updates, and gradients match BatchNorm."""
    torch.manual_seed(7)
    reference = batch_norm_factory(3)
    safe_batch_norm = safe_batch_norm_factory(3)
    safe_batch_norm.load_state_dict(reference.state_dict())
    reference_input = torch.randn(input_shape, requires_grad=True)
    protected_input = reference_input.detach().clone().requires_grad_()

    expected = reference(reference_input)
    actual = safe_batch_norm(protected_input)
    expected.square().sum().backward()
    actual.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(protected_input.grad, reference_input.grad)
    torch.testing.assert_close(safe_batch_norm.weight.grad, reference.weight.grad)
    torch.testing.assert_close(safe_batch_norm.bias.grad, reference.bias.grad)
    _assert_running_stats_equal(safe_batch_norm, reference)


@pytest.mark.parametrize(
    ("batch_norm_factory", "safe_batch_norm_factory", "input_shape"),
    BATCH_NORM_CASES,
)
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_forward_preserves_running_stats(
    batch_norm_factory: BatchNormType,
    safe_batch_norm_factory: BatchNormType,
    input_shape: tuple[int, ...],
    non_finite: float,
) -> None:
    """Verify a faulty input cannot change persistent statistics."""
    safe_batch_norm = safe_batch_norm_factory(3)
    expected_state = deepcopy(safe_batch_norm)
    input = torch.randn(input_shape)
    input.flatten()[0] = non_finite

    with pytest.raises(FloatingPointError, match="non-finite input"):
        safe_batch_norm(input)

    _assert_running_stats_equal(safe_batch_norm, expected_state)
    assert isinstance(safe_batch_norm, batch_norm_factory)


def test_eval_forward_rejects_non_finite_input() -> None:
    """Verify evaluation also fails at the non-finite input boundary."""
    safe_batch_norm = SafeBatchNorm2d(3)
    safe_batch_norm.eval()
    expected_state = deepcopy(safe_batch_norm)
    input = torch.full((2, 3, 4, 4), float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite input"):
        safe_batch_norm(input)

    _assert_running_stats_equal(safe_batch_norm, expected_state)


def test_forward_without_running_stats_rejects_non_finite_input() -> None:
    """Verify the input guard does not depend on persistent statistics."""
    safe_batch_norm = SafeBatchNorm2d(3, track_running_stats=False)
    input = torch.full((2, 3, 4, 4), float("inf"))

    with pytest.raises(FloatingPointError, match="non-finite input"):
        safe_batch_norm(input)

    assert safe_batch_norm.running_mean is None
    assert safe_batch_norm.running_var is None
    assert safe_batch_norm.num_batches_tracked is None


def test_state_dict_matches_native_batch_norm() -> None:
    """Verify inheritance preserves native BatchNorm state-dict keys."""
    safe_keys = SafeBatchNorm2d(3).state_dict().keys()
    native_keys = nn.BatchNorm2d(3).state_dict().keys()
    assert safe_keys == native_keys
