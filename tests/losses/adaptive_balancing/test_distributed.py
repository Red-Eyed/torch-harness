"""Two-rank tests through real DDP forward, measurement, and backward."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from torch_harness import BalancerConfig, LossBalancer, LossTarget, weighted_loss


def _distributed_worker(rank: int, rendezvous: str) -> None:
    """Check global statistics and backward gradients with unequal local losses."""
    distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        balancer = LossBalancer(
            "main",
            {"a": LossTarget(0.3)},
            config=BalancerConfig(
                update_every=2, ema_decay=0, adaptation_rate=1, max_log_change=100
            ),
        )
        for step in range(6):
            optimizer.zero_grad()
            shared = ddp(torch.ones(1, 1))
            scale = float(rank + 1)
            losses = {"main": shared.sum(), "a": scale * shared.sum()}
            if step == 2 and rank == 1:
                losses.pop("a")
            weights = balancer.update(losses, shared, step)
            assert weights["a"] == pytest.approx(0.2)
            assert weights["main"] == 1
            assert model.weight.grad is None
            if step == 2:
                assert balancer.skipped_updates == 1
                assert balancer.state_dict()["ema"]["a"] == 1.5
            weighted_loss(losses, weights).backward()
            expected = 1.1 if step == 2 else 1.3
            torch.testing.assert_close(model.weight.grad, torch.tensor([[expected]]))
            optimizer.step()
        # Identical reduced statistics lead to identical checkpoint state on both ranks.
        state = torch.tensor([balancer.weights["a"], balancer.skipped_updates])
        other = state.clone()
        distributed.broadcast(other, src=0)
        torch.testing.assert_close(state, other)
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_gloo_available(), reason="Gloo unavailable")
def test_ddp_consistency_and_normal_backward(tmp_path: Path) -> None:
    """Preserve real DDP synchronization while balancing uneven rank gradients."""
    rendezvous = (tmp_path / "rendezvous").as_uri()
    multiprocessing.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=True)
