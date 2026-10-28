"""Analytical and real-lifecycle checks for training FLOP measurement."""

from pathlib import Path
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode
from typing_extensions import override

from torch_harness.flops import FlopReport, UnmeasuredFlops
from torch_harness.lightning import FlopCounter


class _Model(LightningModule):
    """Use one bias-free matrix multiplication with analytically known work."""

    def __init__(
        self, *, manual: bool = False, fail: bool = False, recompute: bool = False
    ) -> None:
        """Choose optimization mode and an optional intentional failure."""
        super().__init__()
        self.layer = nn.Linear(4, 2, bias=False)
        self.automatic_optimization = not manual
        self.fail = fail
        self.recompute = recompute

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Execute forward and optionally drive Lightning's manual backward."""
        batch = cast(tuple[torch.Tensor], args[0])
        if self.recompute:
            loss = checkpoint(
                self.layer, batch[0].requires_grad_(), use_reentrant=True
            ).sum()
        else:
            loss = self.layer(batch[0]).sum()
        if self.fail:
            raise RuntimeError("intentional failure")
        if not self.automatic_optimization:
            optimizer = self.optimizers()
            assert not isinstance(optimizer, list)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use SGD without extra registered matrix operations."""
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.fixture
def loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    """Provide three deterministic microbatches."""
    return DataLoader(TensorDataset(torch.ones(3, 4)), batch_size=1)


def _trainer(
    tmp_path: Path, *callbacks: FlopCounter, accumulate_grad_batches: int = 1
) -> Trainer:
    """Create an isolated CPU trainer without external logging or artifacts."""
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=list(callbacks),
        accumulate_grad_batches=accumulate_grad_batches,
    )


@pytest.mark.parametrize(
    ("manual", "accumulation"), [(False, 1), (False, 2), (True, 1)]
)
def test_real_training_counts(
    tmp_path: Path,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    manual: bool,
    accumulation: int,
) -> None:
    """Count actual forward and weight-gradient matmuls under both optimizers."""
    callback = FlopCounter(max_batches=3, log_metrics=False)
    trainer = _trainer(tmp_path, callback, accumulate_grad_batches=accumulation)
    trainer.fit(_Model(manual=manual), train_dataloaders=loader)
    report = callback.latest_report
    assert isinstance(report, FlopReport)
    assert report.batch_flops == 32
    assert report.backward_flops == 16
    assert report.non_backward_flops == 16
    assert callback._counter is None


def test_failure_cleans_up(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Propagate failures without leaving a dispatch mode or partial report."""
    callback = FlopCounter(log_metrics=False)
    with pytest.raises(RuntimeError, match="intentional failure"):
        _trainer(tmp_path, callback).fit(_Model(fail=True), train_dataloaders=loader)
    assert isinstance(callback.latest_report, UnmeasuredFlops)
    assert callback._counter is None
    with FlopCounterMode(display=False) as counter:
        torch.ones(1, 4) @ torch.ones(4, 2)
    assert counter.get_total_flops() == 16


@pytest.mark.parametrize("window", [2, 4, 8])
def test_partitioned_attention(window: int) -> None:
    """Window tensor shapes capture the linear reduction in attention work."""
    tokens, width = 8, 4
    q = torch.ones(tokens // window, window, width)
    with FlopCounterMode(display=False) as counter:
        scores = q @ q.mT
        scores @ q
    assert counter.get_total_flops() == 4 * tokens * window * width


@pytest.mark.parametrize(("interval", "maximum"), [(0, 1), (1, 0), (-1, 1)])
def test_invalid_sampling(interval: int, maximum: int) -> None:
    """Reject invalid sampling without creating a dispatch mode."""
    with pytest.raises(ValueError, match="positive"):
        FlopCounter(every_n_batches=interval, max_batches=maximum)


def test_sampling_and_custom_formula(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Apply local formulas only to selected batches without global registration."""

    def formula(*args: object, **kwargs: object) -> int:
        """Assign a recognizable constant to every matrix multiplication."""
        return 7

    callback = FlopCounter(
        every_n_batches=2,
        max_batches=2,
        custom_mapping={torch.ops.aten.mm: formula},
        log_metrics=False,
    )
    trainer = _trainer(tmp_path, callback)
    trainer.fit(_Model(), train_dataloaders=loader)
    assert callback._measured_batches == 2
    assert isinstance(callback.latest_report, FlopReport)
    assert callback.latest_report.batch_flops == 14
    with FlopCounterMode(display=False) as counter:
        torch.ones(1, 4) @ torch.ones(4, 2)
    assert counter.get_total_flops() == 16


def test_guard_rejects_higher_order_execution() -> None:
    """Fail before a backend can evaluate extra branches or skip kernels."""
    from torch_harness.flops._guard import ExecutionGuard

    def branch(x: torch.Tensor) -> torch.Tensor:
        """Provide a valid branch that must never execute under the guard."""
        raise AssertionError("branch executed")

    with (
        pytest.raises(RuntimeError, match="higher-order"),
        FlopCounterMode(display=False),
        ExecutionGuard(),
    ):
        torch.ops.higher_order.cond(
            torch.tensor(True), branch, branch, (torch.ones(1),)
        )


def test_checkpoint_recomputation(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Attribute recomputed forward arithmetic to its enclosing backward call."""
    callback = FlopCounter(log_metrics=False)
    _trainer(tmp_path, callback).fit(_Model(recompute=True), train_dataloaders=loader)
    report = callback.latest_report
    assert isinstance(report, FlopReport)
    assert report.batch_flops == 64
    assert report.backward_flops == 48


def test_logging_and_preservation(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Keep updates identical to baseline and expose rank-local logger metrics."""
    from lightning.pytorch.loggers import CSVLogger

    measured = _Model()
    baseline = _Model()
    baseline.load_state_dict(measured.state_dict())
    callback = FlopCounter(max_batches=3)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=1,
        logger=CSVLogger(tmp_path),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
        log_every_n_steps=1,
    )
    trainer.fit(measured, train_dataloaders=loader)
    _trainer(tmp_path).fit(baseline, train_dataloaders=loader)
    torch.testing.assert_close(
        measured.layer.weight, baseline.layer.weight, rtol=0, atol=0
    )
    assert trainer.callback_metrics["flops/batch"].item() == 32
    assert trainer.callback_metrics["flops/backward"].item() == 16


@pytest.mark.parametrize("compilation", ["submodule", "lightning", "inplace"])
def test_compiled_module_is_rejected(tmp_path: Path, compilation: str) -> None:
    """Reject compiled modules before executing a potentially opaque graph."""
    callback = FlopCounter()
    model = _Model()
    match compilation:
        case "submodule":
            model.layer = cast(nn.Linear, torch.compile(model.layer, backend="eager"))
        case "lightning":
            from lightning.pytorch.utilities.compile import from_compiled
            from torch._dynamo.eval_frame import OptimizedModule

            compiled = cast(OptimizedModule, torch.compile(model, backend="eager"))
            model = cast(_Model, from_compiled(compiled))
        case "inplace":
            model.layer.compile(backend="eager")
    with pytest.raises(RuntimeError, match="uncompiled"):
        callback.on_fit_start(_trainer(tmp_path, callback), model)
