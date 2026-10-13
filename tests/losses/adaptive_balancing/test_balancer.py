"""Numerical and lifecycle tests for gradient-based auxiliary balancing."""

from io import BytesIO
from math import exp
from typing import TypedDict

import pytest
import torch

from torch_harness import BalancerConfig, LossBalancer, LossTarget, weighted_loss
from torch_harness.losses.adaptive_balancing import Unavailable
from torch_harness.losses.adaptive_balancing.measurement import measure_gradient
from torch_harness.losses.adaptive_balancing.numerics import smooth_norm, target_weight


class ConfigOverrides(TypedDict, total=False):
    """Describe invalid test inputs without broadening the constructor contract."""

    update_every: int
    ema_decay: float
    adaptation_rate: float
    max_log_change: float
    gradient_floor: float


@pytest.fixture
def immediate_config() -> BalancerConfig:
    """Provide exact single-update adaptation for deterministic arithmetic."""
    return BalancerConfig(
        update_every=1, ema_decay=0, adaptation_rate=1, max_log_change=100
    )


def test_gradient_norm_and_no_accumulation() -> None:
    """Measure a group norm while preserving existing gradients and the graph."""
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    x.grad = torch.ones_like(x)
    loss = x.square().sum() + 12 * y
    sample = measure_gradient(loss, (x, y))
    assert sample.status == "valid"
    assert sample.norm == pytest.approx((100 + 144) ** 0.5)
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert y.grad is None
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([7.0, 9.0]))


def test_multiple_auxiliary_and_fixed(immediate_config: BalancerConfig) -> None:
    """Set independent gradient ratios while preserving primary and fixed weights."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    losses = {"main": x.sum(), "a": 4 * x.sum(), "b": 2 * x.sum(), "fixed": 9 * x.sum()}
    balancer = LossBalancer(
        "main",
        {"a": LossTarget(0.3), "b": LossTarget(0.2)},
        fixed={"fixed": 0.5},
        config=immediate_config,
    )
    weights = balancer.update(losses, x, 0)
    assert weights == pytest.approx({"main": 1, "a": 0.075, "b": 0.1, "fixed": 0.5})
    weighted_loss(losses, weights).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 6.0))
    assert balancer.diagnostics["a"].gradient_ratio == pytest.approx(0.3)
    assert isinstance(balancer.diagnostics["fixed"].gradient_norm, Unavailable)


def test_ema_and_smooth_updates() -> None:
    """Verify EMA initialization and bounded logarithmic motion analytically."""
    assert smooth_norm(0, 5, 0.9) == 5
    assert smooth_norm(5, 15, 0.9) == pytest.approx(6)
    balancer = LossBalancer(
        "main",
        {"a": LossTarget(0.3)},
        config=BalancerConfig(
            update_every=1, ema_decay=0.5, adaptation_rate=0.5, max_log_change=0.2
        ),
    )
    x = torch.tensor(1.0, requires_grad=True)
    weights = balancer.update({"main": 10 * x, "a": x}, x, 0)
    assert weights["a"] == pytest.approx(0.1 * exp(0.2))
    balancer.update({"main": 20 * x, "a": 3 * x}, x, 1)
    assert balancer.diagnostics["main"].ema_norm == pytest.approx(15)
    assert balancer.diagnostics["a"].ema_norm == pytest.approx(2)


@pytest.mark.parametrize(
    ("reference", "auxiliary", "expected"), [(1e10, 1, 2), (1, 1e10, 0.01)]
)
def test_bounds(
    reference: float,
    auxiliary: float,
    expected: float,
    immediate_config: BalancerConfig,
) -> None:
    """Clamp extreme feasible ratios to configured bounds."""
    target = LossTarget(0.3, min_weight=0.01, max_weight=2)
    assert target_weight(reference, auxiliary, target) == pytest.approx(expected)
    x = torch.tensor(1.0, requires_grad=True)
    balancer = LossBalancer("main", {"a": target}, config=immediate_config)
    assert balancer.update({"main": reference * x, "a": auxiliary * x}, x, 0)[
        "a"
    ] == pytest.approx(expected)
    assert balancer.diagnostics["a"].saturated


@pytest.mark.parametrize(
    "mode", ["zero", "tiny", "disconnected", "detached", "nan", "inf"]
)
def test_invalid_auxiliary_freezes_state(
    mode: str, immediate_config: BalancerConfig
) -> None:
    """Avoid escalation and report a reason for unusable auxiliary gradients."""
    x = torch.tensor(1.0, requires_grad=True)
    match mode:
        case "zero":
            auxiliary = x * 0
        case "tiny":
            auxiliary = x * 1e-12
        case "disconnected":
            auxiliary = torch.tensor(1.0, requires_grad=True) * 2
        case "detached":
            auxiliary = torch.tensor(1.0)
        case "nan":
            auxiliary = x * float("nan")
        case _:
            auxiliary = x * float("inf")
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    assert balancer.update({"main": x, "a": auxiliary}, x, 0)["a"] == 0.1
    assert balancer.skipped_updates == 1
    assert balancer.state_dict()["ema"]["a"] == 0
    assert balancer.diagnostics["a"].status != "valid"


def test_vanishing_reference_freezes_all(immediate_config: BalancerConfig) -> None:
    """Avoid decreasing auxiliary weights using an unusable reference gradient."""
    x = torch.tensor(1.0, requires_grad=True)
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    assert balancer.update({"main": x * 0, "a": x}, x, 0)["a"] == 0.1
    assert balancer.diagnostics["a"].status == "reference_tiny"
    assert balancer.state_dict()["ema"] == {"main": 0, "a": 0}


def test_frequency_absence_and_reappearance() -> None:
    """Skip duplicate and off-cadence steps and resume optional losses safely."""
    x = torch.tensor(1.0, requires_grad=True)
    balancer = LossBalancer(
        "main", {"a": LossTarget(0.3)}, config=BalancerConfig(update_every=2)
    )
    balancer.update({"main": x}, x, 0)
    assert balancer.skipped_updates == 1
    initial = balancer.weights
    balancer.update({"main": x, "a": 2 * x}, x.detach(), 0)
    balancer.update({"main": x, "a": 2 * x}, x.detach(), 1)
    assert balancer.weights == initial
    balancer.update({"main": x, "a": 2 * x}, x, 2)
    assert balancer.weights["a"] > initial["a"]


def test_checkpoint_round_trip_and_continuation() -> None:
    """Restore a torch.save checkpoint and continue with identical controller state."""
    first = LossBalancer(
        "main", {"a": LossTarget(0.3)}, config=BalancerConfig(update_every=1)
    )
    second = LossBalancer(
        "main", {"a": LossTarget(0.3)}, config=BalancerConfig(update_every=1)
    )
    x = torch.tensor(1.0, requires_grad=True)
    first.update({"main": 5 * x, "a": x}, x, 0)
    buffer = BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second.load_state_dict(torch.load(buffer, weights_only=True))
    for step in range(1, 10):
        losses = {"main": 5 * x, "a": (step + 1) * x}
        assert first.update(losses, x, step) == second.update(losses, x, step)
        assert first.state_dict() == second.state_dict()
    snapshot = second.state_dict()
    snapshot["weights"]["a"] = float("nan")
    before = second.state_dict()
    with pytest.raises(ValueError, match="checkpoint weight"):
        second.load_state_dict(snapshot)
    assert second.state_dict() == before


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_precision_and_backward_preservation(
    dtype: torch.dtype, immediate_config: BalancerConfig
) -> None:
    """Compare measured backward against an ordinary explicitly weighted objective."""
    parameter = torch.tensor([1.0, 2.0], dtype=dtype, requires_grad=True)
    shared = parameter * 2
    losses = {"main": shared.square().sum(), "a": 3 * shared.square().sum()}
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    weights = balancer.update(losses, shared, 0)
    assert parameter.grad is None
    weighted_loss(losses, weights).backward()
    torch.testing.assert_close(
        parameter.grad, 8 * parameter.detach() * 1.3, rtol=0.01, atol=0.01
    )


def test_synthetic_optimization_converges() -> None:
    """Approach two configured ratios throughout a deterministic optimization run."""
    parameter = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    balancer = LossBalancer(
        "main",
        {"a": LossTarget(0.3), "b": LossTarget(0.2)},
        config=BalancerConfig(update_every=1, ema_decay=0.8, adaptation_rate=0.2),
    )
    for step in range(100):
        optimizer.zero_grad()
        shared = parameter * 1.0
        base = shared.square()
        losses = {"main": base, "a": 4 * base, "b": 2 * base}
        weights = balancer.update(losses, shared, step)
        weighted_loss(losses, weights).backward()
        optimizer.step()
    assert balancer.diagnostics["a"].gradient_ratio == pytest.approx(0.3, rel=1e-5)
    assert balancer.diagnostics["b"].gradient_ratio == pytest.approx(0.2, rel=1e-5)
    assert abs(float(parameter.detach())) < 0.2


def test_zero_weight_omits_invalid_loss() -> None:
    """A disabled fixed term cannot poison the aggregated objective."""
    x = torch.tensor(2.0, requires_grad=True)
    total = weighted_loss(
        {"main": x, "disabled": x * float("nan")}, {"main": 1, "disabled": 0}
    )
    total.backward()
    assert float(total.detach()) == 2
    assert x.grad == 1


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_target_configuration(ratio: float) -> None:
    """Reject invalid target influence before constructing the controller."""
    with pytest.raises(ValueError, match="positive"):
        LossTarget(ratio)


def test_configuration_and_input_errors() -> None:
    """Reject ambiguous names, invalid inputs, and incompatible checkpoints."""
    with pytest.raises(ValueError, match="disjoint"):
        LossBalancer("main", {"main": LossTarget(0.3)})
    balancer = LossBalancer("main", {"a": LossTarget(0.3)})
    x = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ValueError, match="reference"):
        balancer.update({"a": x}, x, 0)
    with pytest.raises(ValueError, match="unconfigured"):
        balancer.update({"main": x, "unknown": x}, x, 0)
    with pytest.raises(ValueError, match="scalar"):
        balancer.update({"main": x.reshape(1)}, x, 0)
    different = LossBalancer("main", {"a": LossTarget(0.4)})
    with pytest.raises(ValueError, match="configuration"):
        balancer.load_state_dict(different.state_dict())


def test_metrics_are_detached_scalars(immediate_config: BalancerConfig) -> None:
    """Expose defined numeric diagnostics without tensors or graph retention."""
    x = torch.tensor(1.0, requires_grad=True)
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    balancer.update({"main": x, "a": 2 * x}, x, 0)
    assert balancer.metrics()["balancer/a/gradient_ratio"] == pytest.approx(0.3)
    assert all(isinstance(value, float) for value in balancer.metrics().values())


@pytest.mark.parametrize("scale", [1e-30, 1e30, 1e38])
def test_scaled_norm_handles_extreme_finite_gradients(scale: float) -> None:
    """Preserve norms whose raw squares would underflow or overflow float32."""
    x = torch.zeros(2, requires_grad=True)
    sample = measure_gradient((scale * x).sum(), (x,))
    assert sample.status == "valid"
    assert sample.norm == pytest.approx(2**0.5 * scale, rel=1e-6)


def test_nonfinite_gradient_with_finite_loss(immediate_config: BalancerConfig) -> None:
    """Reject an infinite derivative even when its scalar objective is finite."""
    x = torch.tensor(0.0, requires_grad=True)
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    assert balancer.update({"main": x, "a": x.sqrt()}, x, 0)["a"] == 0.1
    assert balancer.diagnostics["a"].status == "nonfinite"


def test_parameter_group_and_partial_connectivity(
    immediate_config: BalancerConfig,
) -> None:
    """Support local parameter measurement with unused parameters in the group."""
    used = torch.nn.Parameter(torch.tensor(1.0))
    unused = torch.nn.Parameter(torch.tensor(2.0))
    balancer = LossBalancer("main", {"a": LossTarget(0.3)}, config=immediate_config)
    weights = balancer.update({"main": used, "a": 2 * used}, (used, unused), 0)
    assert weights["a"] == pytest.approx(0.15)
    assert used.grad is None
    assert unused.grad is None


def test_no_adaptive_losses_require_no_gradient_measurement() -> None:
    """Use only reference and fixed terms without requiring differentiable targets."""
    balancer = LossBalancer("main", {}, fixed={"fixed": 0.5})
    weights = balancer.update({"main": torch.tensor(2.0)}, torch.tensor(0.0), 0)
    assert weights == {"main": 1, "fixed": 0.5}


@pytest.mark.parametrize(
    "field", ["version", "weights", "ema", "last_step", "skipped_updates"]
)
def test_malformed_checkpoints_are_atomic(field: str) -> None:
    """Reject invalid boundary data without partially replacing controller state."""
    balancer = LossBalancer("main", {"a": LossTarget(0.3)})
    before = balancer.state_dict()
    invalid: dict[str, object] = dict(before)
    invalid[field] = "invalid"
    with pytest.raises(ValueError, match="checkpoint"):
        balancer.load_state_dict(invalid)
    assert balancer.state_dict() == before


@pytest.mark.parametrize(
    "config",
    [
        {"update_every": 0},
        {"update_every": True},
        {"ema_decay": 1.0},
        {"adaptation_rate": 0.0},
        {"max_log_change": float("inf")},
        {"gradient_floor": 0.0},
    ],
)
def test_invalid_controller_configuration(config: ConfigOverrides) -> None:
    """Validate cadence and numerical policy before use."""
    with pytest.raises((ValueError, TypeError), match="must"):
        BalancerConfig(**config)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), 0.0])
def test_invalid_aggregation_weights(weight: float) -> None:
    """Reject invalid weights or an objective with no active terms."""
    with pytest.raises(ValueError, match=r"weights|positive weight"):
        weighted_loss({"main": torch.tensor(1.0)}, {"main": weight})


def test_invalid_targets_and_steps() -> None:
    """Reject invalid target ownership and backwards controller steps."""
    x = torch.tensor(1.0, requires_grad=True)
    balancer = LossBalancer("main", {"a": LossTarget(0.3)})
    with pytest.raises(ValueError, match="unique"):
        balancer.update({"main": x}, (x, x), 0)
    with pytest.raises(ValueError, match="gradients"):
        balancer.update({"main": x}, x.detach(), 0)
    balancer.update({"main": x}, x, 1)
    with pytest.raises(ValueError, match="backwards"):
        balancer.update({"main": x}, x, 0)
