"""Validated policy for gradient-based auxiliary loss balancing."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LossTarget:
    """Specify relative gradient influence and strictly positive weight bounds."""

    ratio: float
    initial_weight: float = 0.1
    min_weight: float = 1e-4
    max_weight: float = 10.0

    def __post_init__(self) -> None:
        """Reject nonfinite values and inconsistent positive bounds."""
        values = (self.ratio, self.initial_weight, self.min_weight, self.max_weight)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("target values must be finite and positive")
        if not self.min_weight <= self.initial_weight <= self.max_weight:
            raise ValueError("initial_weight must lie within weight bounds")


@dataclass(frozen=True)
class BalancerConfig:
    """Control measurement cadence, EMA decay, and log-space adaptation."""

    update_every: int = 10
    ema_decay: float = 0.9
    adaptation_rate: float = 0.1
    max_log_change: float = 0.2
    gradient_floor: float = 1e-8
    synchronize: bool = True

    def __post_init__(self) -> None:
        """Validate numerical policy before any training graph is inspected."""
        if isinstance(self.update_every, bool) or not isinstance(
            self.update_every, int
        ):
            raise TypeError("update_every must be an integer")
        if self.update_every < 1:
            raise ValueError("update_every must be positive")
        if not isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not isfinite(self.adaptation_rate) or not 0 < self.adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be in (0, 1]")
        if not all(
            isfinite(value) and value > 0
            for value in (self.max_log_change, self.gradient_floor)
        ):
            raise ValueError("max_log_change and gradient_floor must be positive")
