"""Analytical and lifecycle checks for one-shot forward FLOP measurement."""

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
    """Expose known forward arithmetic followed by additional loss arithmetic."""

    def __init__(
        self,
        *,
        manual: bool = False,
        fail: bool = False,
        recompute: bool = False,
        keyword: bool = False,
        inner: bool = False,
    ) -> None:
        """Configure optimization, checkpointing, and model invocation variants."""
        super().__init__()
        self.layer = nn.Linear(4, 2, bias=False)
        self.automatic_optimization = not manual
        self.fail = fail
        self.recompute = recompute
        self.keyword = keyword
        self.inner = inner

    @override
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Run one matrix multiplication or an equivalent checkpointed forward."""
        inputs = cast(torch.Tensor, args[0] if args else kwargs["inputs"])
        if self.recompute:
            output = checkpoint(self.layer, inputs.requires_grad_(), use_reentrant=True)
        else:
            output = self.layer(inputs)
        if self.fail:
            raise RuntimeError("intentional failure")
        return output

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Execute extra model and loss matmuls that must not enter the report."""
        batch = cast(tuple[torch.Tensor], args[0])
        if self.inner:
            prediction = self.layer(batch[0])
        elif self.keyword:
            prediction = self(inputs=batch[0])
        else:
            prediction = self(batch[0])
        extra_prediction = self.layer(batch[0]) if self.inner else self(batch[0])
        loss = (
            ((prediction + extra_prediction) @ prediction.new_ones(2, 2))
            .square()
            .mean()
        )
        if not self.automatic_optimization:
            optimizer = self.optimizers()
            assert not isinstance(optimizer, list)
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
        return loss

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use SGD to compare model updates against an uninstrumented baseline."""
        return torch.optim.SGD(self.parameters(), lr=0.001)


@pytest.fixture
def loader() -> DataLoader[tuple[torch.Tensor, ...]]:
    """Provide two full batches followed by a differently shaped last batch."""
    return DataLoader(TensorDataset(torch.ones(5, 4)), batch_size=2)


def _trainer(
    tmp_path: Path, *callbacks: FlopCounter, accumulate_grad_batches: int = 1
) -> Trainer:
    """Create a CPU trainer spanning multiple epochs without external artifacts."""
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=list(callbacks),
        accumulate_grad_batches=accumulate_grad_batches,
    )


@pytest.mark.parametrize(
    ("manual", "accumulation", "keyword", "recompute"),
    [
        (False, 1, False, False),
        (False, 2, True, False),
        (True, 1, False, False),
        (False, 1, False, True),
    ],
)
def test_only_first_forward_counts(
    tmp_path: Path,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    manual: bool,
    accumulation: int,
    keyword: bool,
    recompute: bool,
) -> None:
    """Exclude later calls, loss, backward, recomputation, batches, and epochs."""
    callback = FlopCounter(log_metrics=False)
    trainer = _trainer(tmp_path, callback, accumulate_grad_batches=accumulation)
    trainer.fit(_Model(manual=manual, keyword=keyword, recompute=recompute), loader)
    report = callback.latest_report
    assert isinstance(report, FlopReport)
    assert report.forward_flops == 32
    assert report.input_shapes == ((2, 4),)
    assert report.batch_size == 2
    assert report.multiply_add_flops == 2
    assert callback._counter is None
    assert not callback._hooks


def test_failure_cleans_up(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Preserve forward exceptions without leaving counting or module hooks."""
    callback = FlopCounter(log_metrics=False)
    with pytest.raises(RuntimeError, match="intentional failure"):
        _trainer(tmp_path, callback).fit(_Model(fail=True), loader)
    assert isinstance(callback.latest_report, UnmeasuredFlops)
    assert callback._counter is None
    assert not callback._hooks
    with FlopCounterMode(display=False) as counter:
        torch.ones(1, 4) @ torch.ones(4, 2)
    assert counter.get_total_flops() == 16


@pytest.mark.parametrize("window", [2, 4, 8])
def test_partitioned_attention(window: int) -> None:
    """Window tensor shapes capture the linear reduction in attention work."""
    q = torch.ones(8 // window, window, 4)
    with FlopCounterMode(display=False) as counter:
        scores = q @ q.mT
        scores @ q
    assert counter.get_total_flops() == 4 * 8 * window * 4


def test_custom_formula_runs_once(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Invoke formulas only during the first forward and leave the registry intact."""
    calls: list[int] = []

    def formula(*args: object, **kwargs: object) -> int:
        """Record measured calls and assign an unmistakable FLOP count."""
        calls.append(1)
        return 7

    callback = FlopCounter(
        custom_mapping={torch.ops.aten.mm: formula}, log_metrics=False
    )
    _trainer(tmp_path, callback).fit(_Model(), loader)
    assert calls == [1]
    assert isinstance(callback.latest_report, FlopReport)
    assert callback.latest_report.forward_flops == 7
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


def test_logging_and_preservation(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Keep updates identical to baseline and log a single forward metric row."""
    import csv

    from lightning.pytorch.loggers import CSVLogger

    measured = _Model()
    baseline = _Model()
    baseline.load_state_dict(measured.state_dict())
    callback = FlopCounter()
    metric_logger = CSVLogger(tmp_path)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=2,
        logger=metric_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
        log_every_n_steps=1,
    )
    trainer.fit(measured, loader)
    _trainer(tmp_path).fit(baseline, loader)
    torch.testing.assert_close(
        measured.layer.weight, baseline.layer.weight, rtol=0, atol=0
    )
    with (Path(metric_logger.log_dir) / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len([row for row in rows if row.get("flops/forward")]) == 1
    assert float(rows[0]["flops/forward"]) == 32
    assert all("flops/backward" not in row for row in rows)


def test_inner_model_and_fit_reset(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Support explicit model selection and one new measurement on a later fit."""
    model = _Model(inner=True)
    callback = FlopCounter(module=model.layer, log_metrics=False)
    _trainer(tmp_path, callback).fit(model, loader)
    first = callback.latest_report
    assert isinstance(first, FlopReport)
    assert first.forward_flops == 32
    _trainer(tmp_path, callback).fit(model, loader)
    assert isinstance(callback.latest_report, FlopReport)
    assert callback.latest_report == first
    assert callback.latest_report is not first


def test_missing_model_call_does_not_count_training_step(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Do not substitute a whole training step when the selected forward is absent."""
    callback = FlopCounter(log_metrics=False)
    _trainer(tmp_path, callback).fit(_Model(inner=True), loader)
    assert isinstance(callback.latest_report, UnmeasuredFlops)
    assert not callback._hooks


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


def test_sanity_validation_is_not_measured(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Ignore earlier validation inputs and retain the first training input shape."""

    class _ValidatedModel(_Model):
        """Run the same forward with smaller inputs during sanity checking."""

        @override
        def validation_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
            """Exercise model calls before and after the measured training batch."""
            batch = cast(tuple[torch.Tensor], args[0])
            return self(batch[0]).sum()

    validation = DataLoader(TensorDataset(torch.ones(2, 4)), batch_size=1)
    callback = FlopCounter(log_metrics=False)
    _trainer(tmp_path, callback).fit(_ValidatedModel(), loader, validation)
    assert isinstance(callback.latest_report, FlopReport)
    assert callback.latest_report.input_shapes == ((2, 4),)
    assert callback.latest_report.forward_flops == 32


def test_nested_input_metadata(
    tmp_path: Path, loader: DataLoader[tuple[torch.Tensor, ...]]
) -> None:
    """Record nested positional and keyword tensors without retaining inputs."""

    class _NestedModel(_Model):
        """Accept a nested sequence and a keyword tensor inside forward."""

        @override
        def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
            """Unpack nested inputs before executing the standard projection."""
            nested = cast(list[tuple[torch.Tensor]], args[0])
            mask = cast(torch.Tensor, kwargs["mask"])
            return self.layer(nested[0][0]) * mask

        @override
        def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
            """Supply nested inputs and leave loss computation outside forward."""
            batch = cast(tuple[torch.Tensor], args[0])
            return self([(batch[0],)], mask=batch[0][:, :2]).sum()

    callback = FlopCounter(log_metrics=False)
    _trainer(tmp_path, callback).fit(_NestedModel(), loader)
    assert isinstance(callback.latest_report, FlopReport)
    assert callback.latest_report.input_shapes == ((2, 4), (2, 2))
    assert callback.latest_report.batch_size == 2
    assert callback.latest_report.forward_flops == 32
