"""Stateful gradient-ratio controller and graph-preserving loss aggregation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Literal, TypedDict

from torch import Tensor

from .config import BalancerConfig, LossTarget
from .measurement import (
    GradientMeasurement,
    GradientStatus,
    measure_losses,
    measurement_targets,
)
from .numerics import adapt_weight, smooth_norm, target_weight

_DEFAULT_CONFIG = BalancerConfig()
UpdateStatus = (
    GradientStatus
    | Literal[
        "not_measured",
        "reference_absent",
        "reference_disconnected",
        "reference_nonfinite",
        "reference_tiny",
    ]
)


@dataclass(frozen=True)
class Unavailable:
    """Explain why a diagnostic has no numerical value."""

    reason: str


@dataclass(frozen=True)
class LossDiagnostics:
    """Detached metrics from the latest call; measurements occur only when due."""

    loss: float
    weight: float
    gradient_norm: float | Unavailable
    weighted_gradient_norm: float | Unavailable
    gradient_ratio: float | Unavailable
    ema_norm: float | Unavailable
    saturated: bool
    status: UpdateStatus


class BalancerState(TypedDict):
    """Primitive checkpoint schema with a configuration fingerprint."""

    version: int
    signature: tuple[str | float | int | bool, ...]
    weights: dict[str, float]
    ema: dict[str, float]
    last_step: int
    skipped_updates: int


def _checkpoint_floats(value: object) -> dict[str, float]:
    """Parse a named numerical record once at the checkpoint boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint statistics must be mappings")
    result: dict[str, float] = {}
    for name, number in value.items():
        if (
            not isinstance(name, str)
            or isinstance(number, bool)
            or not isinstance(number, (int, float))
        ):
            raise ValueError("checkpoint statistics must map names to numbers")
        result[name] = float(number)
    return result


def _checkpoint_counter(value: object, minimum: int) -> int:
    """Validate a checkpoint lifecycle counter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("invalid checkpoint counter")
    return value


def _validate_losses(losses: Mapping[str, Tensor], names: set[str]) -> None:
    """Reject unknown names and non-scalar or incompatible loss tensors."""
    if set(losses) - names:
        raise ValueError(f"unconfigured losses: {sorted(set(losses) - names)}")
    devices = {loss.device for loss in losses.values()}
    if len(devices) > 1:
        raise ValueError("losses must use one device")
    for loss in losses.values():
        if loss.ndim != 0 or not loss.is_floating_point():
            raise ValueError("losses must be scalar floating-point tensors")


def weighted_loss(losses: Mapping[str, Tensor], weights: Mapping[str, float]) -> Tensor:
    """Sum named losses with finite nonnegative weights, retaining ordinary gradients.

    Zero-weight terms are omitted, including nonfinite disabled losses. All
    losses must have explicit weights; at least one weight must be positive.
    Low-precision losses are promoted to float32 before aggregation.
    """
    _validate_losses(losses, set(weights))
    total: Tensor | None = None
    for name, loss in losses.items():
        weight = weights[name]
        if not isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
        if weight == 0:
            continue
        value = (
            loss
            if loss.dtype.is_floating_point and loss.element_size() >= 4
            else loss.float()
        )
        term = value * weight
        total = term if total is None else total + term
    if total is None:
        raise ValueError("at least one supplied loss must have a positive weight")
    return total


class LossBalancer:
    """Balance auxiliary gradient influence while keeping the reference weight one.

    Call before backward with unscaled losses and a shared activation, or a
    sequence of shared tensors. Parameters are supported outside distributed
    synchronization. Configuration and step cadence must match across ranks.
    This controller owns no optimizer parameters and retains no training graph.
    """

    def __init__(
        self,
        reference: str,
        adaptive: Mapping[str, LossTarget],
        *,
        fixed: Mapping[str, float] | None = None,
        config: BalancerConfig = _DEFAULT_CONFIG,
    ) -> None:
        """Copy configuration and initialize weights without allocating tensors."""
        fixed_weights = dict(fixed or {})
        if not reference or any(not name for name in (*adaptive, *fixed_weights)):
            raise ValueError("loss names must be nonempty")
        if (
            reference in adaptive
            or reference in fixed_weights
            or set(adaptive) & set(fixed_weights)
        ):
            raise ValueError("reference, adaptive, and fixed names must be disjoint")
        if any(not isfinite(value) or value < 0 for value in fixed_weights.values()):
            raise ValueError("fixed weights must be finite and nonnegative")
        self.reference = reference
        self.config = config
        self._adaptive = dict(sorted(adaptive.items()))
        self._fixed = fixed_weights
        self._weights = {
            name: target.initial_weight for name, target in self._adaptive.items()
        }
        self._ema = dict.fromkeys((reference, *self._adaptive), 0.0)
        self._last_step = -1
        self.skipped_updates = 0
        self._diagnostics: dict[str, LossDiagnostics] = {}

    @property
    def weights(self) -> dict[str, float]:
        """Return an independent snapshot including reference and fixed weights."""
        return {self.reference: 1.0, **self._weights, **self._fixed}

    @property
    def diagnostics(self) -> dict[str, LossDiagnostics]:
        """Return metrics for supplied losses from the latest update call."""
        return dict(self._diagnostics)

    def update(
        self,
        losses: Mapping[str, Tensor],
        shared: Tensor | Sequence[Tensor],
        step: int,
    ) -> dict[str, float]:
        """Measure when due, update valid objectives, and return detached weights.

        Missing auxiliaries retain state for later reappearance. Repeated steps
        do not remeasure, including gradient-accumulation microbatches. Invalid
        measurements freeze weights and EMA rather than escalating weights.
        Invalid scalar loss values remain visible to the caller's fault policy.
        """
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step < self._last_step:
            raise ValueError("step must not move backwards")
        _validate_losses(losses, set(self.weights))
        if self.reference not in losses:
            raise ValueError("reference loss is required")
        measured: dict[str, GradientMeasurement] = {}
        due = step % self.config.update_every == 0 and step != self._last_step
        if due and self._adaptive:
            targets = measurement_targets(shared)
            if targets[0].device != losses[self.reference].device:
                raise ValueError("shared targets and losses must use one device")
            measured = measure_losses(
                losses,
                (self.reference, *self._adaptive),
                targets,
                synchronize=self.config.synchronize,
                floor=self.config.gradient_floor,
            )
            self._advance(measured)
        self._last_step = step
        self._diagnostics = self._build_diagnostics(losses, measured)
        return self.weights

    def _advance(self, measured: Mapping[str, GradientMeasurement]) -> None:
        """Update usable EMA statistics and independently adapt auxiliary weights."""
        reference = measured[self.reference]
        if reference.status != "valid":
            self.skipped_updates += len(self._adaptive)
            return
        self._ema[self.reference] = smooth_norm(
            self._ema[self.reference], reference.norm, self.config.ema_decay
        )
        for name, target in self._adaptive.items():
            sample = measured[name]
            if sample.status != "valid":
                self.skipped_updates += 1
                continue
            self._ema[name] = smooth_norm(
                self._ema[name], sample.norm, self.config.ema_decay
            )
            desired = target_weight(self._ema[self.reference], self._ema[name], target)
            updated = adapt_weight(self._weights[name], desired, self.config)
            self._weights[name] = min(
                target.max_weight, max(target.min_weight, updated)
            )

    def _build_diagnostics(
        self, losses: Mapping[str, Tensor], measured: Mapping[str, GradientMeasurement]
    ) -> dict[str, LossDiagnostics]:
        """Materialize only scalar diagnostics without keeping loss tensors alive."""
        result: dict[str, LossDiagnostics] = {}
        weights = self.weights
        reference = measured.get(self.reference)
        for name, loss in losses.items():
            sample = measured.get(name)
            reason: UpdateStatus = (
                sample.status if sample is not None else "not_measured"
            )
            raw: float | Unavailable = Unavailable(reason)
            weighted: float | Unavailable = Unavailable(reason)
            ratio: float | Unavailable = Unavailable(reason)
            if sample is not None and sample.status in ("valid", "tiny"):
                raw = sample.norm
                weighted = weights[name] * sample.norm
                if reference is not None and reference.status == "valid":
                    ratio = weighted / reference.norm
                else:
                    ratio = Unavailable("reference_unavailable")
            ema = self._ema.get(name, 0.0)
            target = self._adaptive.get(name)
            saturated = target is not None and (
                isclose(weights[name], target.min_weight, rel_tol=1e-12)
                or isclose(weights[name], target.max_weight, rel_tol=1e-12)
            )
            status = reason
            if (
                name in self._adaptive
                and reference is not None
                and reference.status != "valid"
            ):
                match reference.status:
                    case "absent":
                        status = "reference_absent"
                    case "disconnected":
                        status = "reference_disconnected"
                    case "nonfinite":
                        status = "reference_nonfinite"
                    case "tiny":
                        status = "reference_tiny"
            result[name] = LossDiagnostics(
                float(loss.detach()),
                weights[name],
                raw,
                weighted,
                ratio,
                ema if ema > 0 else Unavailable("uninitialized"),
                saturated,
                status,
            )
        return result

    def metrics(self) -> dict[str, float]:
        """Flatten defined numeric diagnostics for TensorBoard or existing loggers."""
        result = {"balancer/skipped_updates": float(self.skipped_updates)}
        for name, sample in self._diagnostics.items():
            prefix = f"balancer/{name}"
            result[f"{prefix}/loss"] = sample.loss
            result[f"{prefix}/weight"] = sample.weight
            result[f"{prefix}/saturated"] = float(sample.saturated)
            result[f"{prefix}/skipped"] = float(
                sample.status not in ("valid", "not_measured")
            )
            values = {
                "gradient_norm": sample.gradient_norm,
                "weighted_gradient_norm": sample.weighted_gradient_norm,
                "gradient_ratio": sample.gradient_ratio,
                "ema_norm": sample.ema_norm,
            }
            for metric, value in values.items():
                match value:
                    case float():
                        result[f"{prefix}/{metric}"] = value
                    case Unavailable():
                        continue
        return result

    def _signature(self) -> tuple[str | float | int | bool, ...]:
        """Fingerprint all policy that affects reproducible checkpoint continuation."""
        values: list[str | float | int | bool] = [
            self.reference,
            self.config.update_every,
            self.config.ema_decay,
            self.config.adaptation_rate,
            self.config.max_log_change,
            self.config.gradient_floor,
            self.config.synchronize,
        ]
        for name, target in self._adaptive.items():
            values.extend(
                (
                    "adaptive",
                    name,
                    target.ratio,
                    target.initial_weight,
                    target.min_weight,
                    target.max_weight,
                )
            )
        for name, weight in sorted(self._fixed.items()):
            values.extend(("fixed", name, weight))
        return tuple(values)

    def state_dict(self) -> BalancerState:
        """Return an independent primitive checkpoint suitable for torch.save."""
        return BalancerState(
            version=1,
            signature=self._signature(),
            weights=dict(self._weights),
            ema=dict(self._ema),
            last_step=self._last_step,
            skipped_updates=self.skipped_updates,
        )

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate a matching checkpoint completely before replacing live state."""
        required = {
            "version",
            "signature",
            "weights",
            "ema",
            "last_step",
            "skipped_updates",
        }
        if set(state) != required:
            raise ValueError("invalid checkpoint fields")
        if (
            type(state["version"]) is not int
            or state["version"] != 1
            or state["signature"] != self._signature()
        ):
            raise ValueError("checkpoint version or configuration does not match")
        weights = _checkpoint_floats(state["weights"])
        ema = _checkpoint_floats(state["ema"])
        if set(weights) != set(self._weights) or set(ema) != set(self._ema):
            raise ValueError("checkpoint loss names do not match")
        for name, value in weights.items():
            target = self._adaptive[name]
            if (
                not isfinite(value)
                or not target.min_weight <= value <= target.max_weight
            ):
                raise ValueError("invalid checkpoint weight")
        if any(not isfinite(value) or value < 0 for value in ema.values()):
            raise ValueError("invalid checkpoint EMA")
        last_step = _checkpoint_counter(state["last_step"], -1)
        skipped_updates = _checkpoint_counter(state["skipped_updates"], 0)
        self._weights = dict(weights)
        self._ema = dict(ema)
        self._last_step = last_step
        self.skipped_updates = skipped_updates
        self._diagnostics = {}
