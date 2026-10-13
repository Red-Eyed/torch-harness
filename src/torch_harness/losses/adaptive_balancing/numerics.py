"""Pure smoothing and bounded gradient-ratio control calculations."""

from math import exp, log

from .config import BalancerConfig, LossTarget


def smooth_norm(previous: float, measured: float, decay: float) -> float:
    """Smooth a valid norm, initializing directly from its first measurement."""
    if previous == 0:
        return measured
    return decay * previous + (1 - decay) * measured


def target_weight(reference: float, auxiliary: float, target: LossTarget) -> float:
    """Calculate a bounded target from positive, validated gradient norms."""
    log_weight = log(target.ratio) + log(reference) - log(auxiliary)
    return exp(min(log(target.max_weight), max(log(target.min_weight), log_weight)))


def adapt_weight(current: float, desired: float, config: BalancerConfig) -> float:
    """Move toward a target smoothly with a per-update multiplicative limit."""
    change = config.adaptation_rate * (log(desired) - log(current))
    change = min(config.max_log_change, max(-config.max_log_change, change))
    return exp(log(current) + change)
