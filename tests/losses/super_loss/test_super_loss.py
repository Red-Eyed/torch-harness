"""Tests for the task-agnostic SuperLoss transformation."""

from collections.abc import Callable
from math import e
from typing import Literal, cast

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from torch_harness import SuperLoss
from torch_harness.losses import SuperLoss as LossesSuperLoss
from torch_harness.losses.super_loss import SuperLoss as ComponentSuperLoss

Reduction = Literal["none", "mean", "sum"]


def test_known_values_and_gradients_match_closed_form() -> None:
    """Verify analytical values and detached-confidence gradient weights."""
    threshold = 1.0
    regularization = 2.0
    task_loss = torch.tensor(
        [threshold - 4.0 / e, threshold, threshold + 4.0 * e],
        dtype=torch.float64,
        requires_grad=True,
    )
    super_loss = SuperLoss(
        threshold=threshold,
        regularization=regularization,
        reduction="none",
    )

    actual = super_loss(task_loss)
    actual.sum().backward()

    torch.testing.assert_close(
        actual,
        torch.tensor([-2.0, 0.0, 6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        task_loss.grad,
        torch.tensor([e, 1.0, 1.0 / e], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        pytest.param("none", torch.tensor([0.0, 3.0]), id="none"),
        pytest.param("mean", torch.tensor(1.5), id="mean"),
        pytest.param("sum", torch.tensor(3.0), id="sum"),
    ],
)
def test_reduction_matches_pytorch_conventions(
    reduction: Reduction,
    expected: Float[Tensor, "..."],
) -> None:
    """Verify all supported reductions operate after itemwise transformation."""
    super_loss = SuperLoss(threshold=0.0, reduction=reduction)
    task_loss = torch.tensor([0.0, e * 2.0])

    actual = super_loss(task_loss)

    torch.testing.assert_close(actual, expected)


def test_confidence_is_capped_for_very_easy_samples() -> None:
    """Verify the branch-point cap limits a sample gradient to Euler's number."""
    task_loss = torch.tensor([-100.0], dtype=torch.float64, requires_grad=True)
    super_loss = SuperLoss(threshold=0.0)

    actual = super_loss(task_loss)
    actual.backward()

    torch.testing.assert_close(task_loss.grad, torch.tensor([e], dtype=torch.float64))


def test_confidence_satisfies_stationary_condition() -> None:
    """Verify confidence solves the optimality equation across the real branch."""
    task_loss = torch.tensor(
        [-2.0 / e, -0.7, -0.2, 0.0, 0.2, 1.0, 10.0, 1e6],
        dtype=torch.float64,
        requires_grad=True,
    )
    super_loss = SuperLoss(threshold=0.0, reduction="sum")

    actual = super_loss(task_loss)
    confidence = torch.autograd.grad(actual, task_loss)[0]
    residual = task_loss.detach() * confidence + 2.0 * confidence.log()

    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_floating_dtype_is_preserved(dtype: torch.dtype) -> None:
    """Verify supported floating dtypes retain their external tensor dtype."""
    task_loss = torch.tensor([0.0, 2.0 * e], dtype=dtype)
    super_loss = SuperLoss(threshold=0.0, reduction="none")

    actual = super_loss(task_loss)

    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual.float(),
        torch.tensor([0.0, 3.0]),
        rtol=1e-2,
        atol=1e-2,
    )


def test_large_finite_loss_remains_finite() -> None:
    """Verify the Lambert W iteration handles the upper float64 range."""
    task_loss = torch.tensor(
        [torch.finfo(torch.float64).max / 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    super_loss = SuperLoss(threshold=0.0)

    actual = super_loss(task_loss)
    actual.backward()

    assert bool(actual.isfinite())
    assert task_loss.grad is not None
    assert bool(task_loss.grad.isfinite().all())


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            lambda: SuperLoss(threshold=float("nan")),
            "threshold must be finite",
            id="threshold",
        ),
        pytest.param(
            lambda: SuperLoss(threshold=0.0, regularization=0.0),
            "regularization must be finite and greater than zero",
            id="zero-regularization",
        ),
        pytest.param(
            lambda: SuperLoss(threshold=0.0, regularization=float("inf")),
            "regularization must be finite and greater than zero",
            id="infinite-regularization",
        ),
    ],
)
def test_invalid_numeric_configuration_is_rejected(
    factory: Callable[[], SuperLoss],
    message: str,
) -> None:
    """Verify invalid numeric configuration fails during construction."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_invalid_reduction_is_rejected() -> None:
    """Verify an unknown reduction fails during construction."""
    with pytest.raises(ValueError, match="unsupported reduction"):
        SuperLoss(threshold=0.0, reduction=cast(Reduction, "median"))


def test_integer_task_loss_is_rejected() -> None:
    """Verify the loss contract rejects tensors that cannot carry gradients."""
    super_loss = SuperLoss(threshold=0.0)

    with pytest.raises(TypeError, match="floating-point"):
        super_loss(torch.tensor([1, 2]))


def test_public_imports_resolve_to_the_component() -> None:
    """Verify the component is exported through both public package levels."""
    assert LossesSuperLoss is ComponentSuperLoss
    assert SuperLoss is ComponentSuperLoss
