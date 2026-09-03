"""Tests for learned uncertainty weighting of peer task losses."""

from math import expm1, log

import pytest
import torch

from torch_harness import UncertaintyWeighting
from torch_harness.losses import UncertaintyWeighting as LossesUncertaintyWeighting
from torch_harness.losses.uncertainty_weighting import (
    UncertaintyWeighting as ComponentUncertaintyWeighting,
)


def _inverse_softplus(value: float) -> float:
    """Return the raw parameter that maps to ``value`` through softplus."""
    return log(expm1(value))


def test_aggregation_matches_known_variances() -> None:
    """Verify the task terms are summed with their inverse variances."""
    weighting = UncertaintyWeighting(num_tasks=2, epsilon=1e-6).double()
    target_variances = torch.tensor([0.5, 2.0], dtype=torch.float64)
    raw_uncertainties = torch.tensor(
        [
            _inverse_softplus(0.5 - weighting.epsilon),
            _inverse_softplus(2.0 - weighting.epsilon),
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        weighting.raw_uncertainties.copy_(raw_uncertainties)

    actual = weighting(torch.tensor([2.0, 6.0], dtype=torch.float64))

    expected = (
        torch.tensor([2.0, 6.0], dtype=torch.float64) / target_variances
        + target_variances.log()
    ).sum()
    torch.testing.assert_close(actual, expected)


def test_task_permutation_does_not_change_aggregation() -> None:
    """Verify no task index receives primary or auxiliary semantics."""
    losses = torch.tensor([0.5, 2.0, 5.0], dtype=torch.float64)
    raw_uncertainties = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    weighting = UncertaintyWeighting(num_tasks=3).double()
    permuted_weighting = UncertaintyWeighting(num_tasks=3).double()
    with torch.no_grad():
        weighting.raw_uncertainties.copy_(raw_uncertainties)
        permuted_weighting.raw_uncertainties.copy_(raw_uncertainties[permutation])

    actual = weighting(losses)
    permuted = permuted_weighting(losses[permutation])

    torch.testing.assert_close(actual, permuted)


def test_effective_weights_start_at_one_and_are_detached() -> None:
    """Verify monitoring weights have unit initialization and no gradient graph."""
    weighting = UncertaintyWeighting(num_tasks=4)

    weights = weighting.effective_weights

    torch.testing.assert_close(weights, torch.ones(4))
    assert not weights.requires_grad
    assert weights.grad_fn is None


def test_optimizer_step_adapts_weights_to_loss_scale() -> None:
    """Verify larger losses receive lower weights after a balancing update."""
    weighting = UncertaintyWeighting(num_tasks=2)
    optimizer = torch.optim.SGD(weighting.parameters(), lr=0.1)
    losses = torch.tensor([0.5, 2.0])

    initial_weights = weighting.effective_weights
    optimizer.zero_grad()
    weighting(losses).backward()
    optimizer.step()
    updated_weights = weighting.effective_weights

    assert updated_weights[0] > initial_weights[0]
    assert updated_weights[1] < initial_weights[1]


def test_gradients_reach_task_losses_and_uncertainties() -> None:
    """Verify backpropagation preserves both required gradient paths."""
    weighting = UncertaintyWeighting(num_tasks=2).double()
    losses = torch.tensor([0.5, 2.0], dtype=torch.float64, requires_grad=True)
    expected_loss_gradients = weighting.effective_weights

    weighting(losses).backward()

    assert losses.grad is not None
    torch.testing.assert_close(losses.grad, expected_loss_gradients)
    assert weighting.raw_uncertainties.grad is not None
    assert bool(weighting.raw_uncertainties.grad.isfinite().all())
    assert bool((weighting.raw_uncertainties.grad != 0.0).all())


@pytest.mark.parametrize(
    ("losses", "message"),
    [
        pytest.param(torch.tensor(1.0), "rank 1", id="scalar"),
        pytest.param(torch.ones(1, 2), "rank 1", id="matrix"),
        pytest.param(torch.ones(2), "contain 3 tasks", id="task-count"),
    ],
)
def test_invalid_loss_shapes_are_rejected(
    losses: torch.Tensor,
    message: str,
) -> None:
    """Verify rank and configured task count are enforced at the boundary."""
    weighting = UncertaintyWeighting(num_tasks=3)

    with pytest.raises(ValueError, match=message):
        weighting(losses)


def test_integer_losses_are_rejected() -> None:
    """Verify losses must use a differentiable floating-point dtype."""
    weighting = UncertaintyWeighting(num_tasks=2)

    with pytest.raises(TypeError, match="floating-point"):
        weighting(torch.tensor([1, 2]))


@pytest.mark.parametrize(
    ("num_tasks", "epsilon", "error", "message"),
    [
        pytest.param(0, 1e-6, ValueError, "greater than zero", id="no-tasks"),
        pytest.param(True, 1e-6, TypeError, "integer", id="boolean-task-count"),
        pytest.param(2, 0.0, ValueError, "between zero and one", id="zero-epsilon"),
        pytest.param(2, 1.0, ValueError, "between zero and one", id="large-epsilon"),
    ],
)
def test_invalid_configuration_is_rejected(
    num_tasks: int,
    epsilon: float,
    error: type[Exception],
    message: str,
) -> None:
    """Verify construction rejects configurations that violate invariants."""
    with pytest.raises(error, match=message):
        UncertaintyWeighting(num_tasks=num_tasks, epsilon=epsilon)


def test_cpu_autocast_keeps_sensitive_operations_in_float32() -> None:
    """Verify mixed-precision inputs are aggregated and differentiated safely."""
    weighting = UncertaintyWeighting(num_tasks=2)
    losses = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = weighting(losses)
    actual.backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor(3.0))
    assert losses.grad is not None
    assert bool(losses.grad.isfinite().all())
    assert weighting.raw_uncertainties.grad is not None
    assert bool(weighting.raw_uncertainties.grad.isfinite().all())


def test_public_imports_resolve_to_the_component() -> None:
    """Verify the component is exported through both public package levels."""
    assert LossesUncertaintyWeighting is ComponentUncertaintyWeighting
    assert UncertaintyWeighting is ComponentUncertaintyWeighting
