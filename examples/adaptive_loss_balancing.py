"""Run deterministic multi-objective optimization with adaptive gradient ratios."""

import torch
from loguru import logger

from torch_harness import BalancerConfig, LossBalancer, LossTarget, weighted_loss


def main() -> None:
    """Train one shared parameter and report convergence using the existing logger."""
    parameter = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    balancer = LossBalancer(
        reference="primary",
        adaptive={
            "auxiliary_a": LossTarget(ratio=0.3),
            "auxiliary_b": LossTarget(ratio=0.2),
        },
        config=BalancerConfig(update_every=1, adaptation_rate=0.2),
    )
    for step in range(100):
        optimizer.zero_grad()
        shared = parameter * 1.0
        base = shared.square()
        losses = {"primary": base, "auxiliary_a": 4 * base, "auxiliary_b": 2 * base}
        weights = balancer.update(losses, shared, step)
        weighted_loss(losses, weights).backward()
        optimizer.step()
        if step % 20 == 0 or step == 99:
            logger.info("Step {}: {}", step, balancer.metrics())
    checkpoint = balancer.state_dict()
    balancer.load_state_dict(checkpoint)


if __name__ == "__main__":
    main()
