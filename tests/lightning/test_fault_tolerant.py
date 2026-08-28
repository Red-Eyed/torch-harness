"""Tests for fault-tolerant Lightning training steps."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from datetime import timedelta
from typing import cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from torch_utils.lightning import FaultTolerantTrainingStep, NonFiniteLossError
from torch_utils.lightning.fault_tolerant import _RollingFaultBudget


class _TrainerStub:
    """Provide the trainer state used by callback setup."""

    def __init__(self, world_size: int = 1) -> None:
        """Store the simulated number of processes."""
        self.world_size = world_size


class _SequenceModel(LightningModule):
    """Return or raise a configured sequence of training-step outcomes."""

    def __init__(self, outcomes: list[STEP_OUTPUT | Exception]) -> None:
        """Store outcomes and a parameter suitable for optimizer construction."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self._outcomes = deque(outcomes)

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Consume the next configured outcome."""
        del args, kwargs
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct the optimizer required by Lightning integration."""
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _IntegrationModel(LightningModule):
    """Produce one non-finite loss followed by one finite loss."""

    def __init__(self) -> None:
        """Initialize a scalar parameter whose update is easy to verify."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Return a bad loss for the first batch and a useful loss thereafter."""
        batch_idx = _extract_batch_index(args, kwargs)
        if batch_idx == 0:
            return self.weight * torch.tensor(float("nan"))
        return self.weight.square()

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct a deterministic SGD optimizer."""
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.fixture
def trainer_stub() -> Trainer:
    """Provide a single-process trainer-shaped object for callback unit tests."""
    return cast(Trainer, _TrainerStub())


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    """Capture rendered Loguru messages for diagnostic assertions."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")
    yield messages
    logger.remove(sink_id)


def _install(
    callback: FaultTolerantTrainingStep,
    model: LightningModule,
    trainer: Trainer,
) -> None:
    """Install a callback through its public Lightning lifecycle hook."""
    callback.setup(trainer, model, "fit")


def _extract_batch_index(args: tuple[object, ...], kwargs: dict[str, object]) -> int:
    """Read Lightning's batch index from a variadic test-model call."""
    if "batch_idx" in kwargs:
        return cast(int, kwargs["batch_idx"])
    return cast(int, args[1])


def test_finite_output_is_returned_unchanged(trainer_stub: Trainer) -> None:
    """Preserve valid training-step output identity."""
    expected = {"loss": torch.tensor(2.0), "metadata": "kept"}
    model = _SequenceModel([expected])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    _install(callback, model, trainer_stub)

    actual = model.training_step(torch.ones(1), 4)

    assert actual is expected


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_loss_is_skipped(
    bad_loss: float,
    trainer_stub: Trainer,
    log_messages: list[str],
) -> None:
    """Convert each non-finite floating-point loss into a skipped batch."""
    model = _SequenceModel([torch.tensor(bad_loss)])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(minutes=30),
    )
    _install(callback, model, trainer_stub)

    output = model.training_step(torch.ones(1), 7)

    assert output is None
    assert any("Skipping training batch 7" in message for message in log_messages)
    assert any("NaN or infinite loss" in message for message in log_messages)


def test_original_exception_is_propagated_when_budget_is_exceeded(
    trainer_stub: Trainer,
) -> None:
    """Re-raise the same exception instance after faults become too frequent."""
    error = RuntimeError("broken batch")
    model = _SequenceModel([error, error])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    _install(callback, model, trainer_stub)

    first_output = model.training_step(torch.ones(1), 0)
    with pytest.raises(RuntimeError) as raised:
        model.training_step(torch.ones(1), 1)

    assert first_output is None
    assert raised.value is error


def test_non_finite_loss_raises_specific_error_when_budget_is_exceeded(
    trainer_stub: Trainer,
) -> None:
    """Expose a typed error when repeated invalid losses exhaust the budget."""
    bad_loss = torch.tensor(float("nan"))
    model = _SequenceModel([bad_loss, bad_loss])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    _install(callback, model, trainer_stub)

    assert model.training_step(torch.ones(1), 0) is None
    with pytest.raises(NonFiniteLossError):
        model.training_step(torch.ones(1), 1)


def test_cuda_oom_is_logged_and_skipped(
    trainer_stub: Trainer,
    log_messages: list[str],
) -> None:
    """Treat a CUDA out-of-memory exception like another tolerated fault."""
    model = _SequenceModel([torch.cuda.OutOfMemoryError("allocation failed")])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    _install(callback, model, trainer_stub)

    output = model.training_step(torch.ones(1), batch_idx=3)

    assert output is None
    assert any("allocation failed" in message for message in log_messages)


def test_fault_budget_forgets_faults_outside_window() -> None:
    """Allow another fault once the previous one leaves the rolling window."""
    budget = _RollingFaultBudget(max_faults=1, window=timedelta(minutes=1))

    first = budget.record(0.0)
    too_soon = budget.record(59.0)
    after_window = budget.record(60.0)

    assert first.tolerated
    assert not too_soon.tolerated
    assert after_window.tolerated


@pytest.mark.parametrize(
    ("max_faults", "fault_window"),
    [
        (0, timedelta(seconds=1)),
        (-1, timedelta(seconds=1)),
        (True, timedelta(seconds=1)),
        (1, timedelta(0)),
        (1, timedelta(seconds=-1)),
    ],
)
def test_invalid_policy_is_rejected(
    max_faults: int,
    fault_window: timedelta,
) -> None:
    """Reject policies that cannot define a useful rolling budget."""
    with pytest.raises(ValueError, match=r"must be .*positive"):
        FaultTolerantTrainingStep(
            max_faults=max_faults,
            fault_window=fault_window,
        )


@pytest.mark.parametrize(("manual", "world_size"), [(True, 1), (False, 2)])
def test_unsafe_training_modes_are_rejected(manual: bool, world_size: int) -> None:
    """Fail before training when a skipped step could corrupt loop state."""
    model = _SequenceModel([torch.tensor(1.0)])
    model.automatic_optimization = not manual
    trainer = cast(Trainer, _TrainerStub(world_size))
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )

    with pytest.raises(RuntimeError):
        callback.setup(trainer, model, "fit")


def test_teardown_restores_training_step(trainer_stub: Trainer) -> None:
    """Remove the instance wrapper so the model's class method is visible again."""
    model = _SequenceModel([torch.tensor(1.0)])
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    _install(callback, model, trainer_stub)

    assert "training_step" in model.__dict__

    callback.teardown(trainer_stub, model, "fit")

    assert "training_step" not in model.__dict__


def test_lightning_skips_bad_batch_before_optimizer_update() -> None:
    """Verify end-to-end that only the finite batch changes model parameters."""
    model = _IntegrationModel()
    callback = FaultTolerantTrainingStep(
        max_faults=1,
        fault_window=timedelta(hours=1),
    )
    dataloader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )

    trainer.fit(model, train_dataloaders=dataloader)

    assert model.weight.item() == pytest.approx(0.8)
    assert "training_step" not in model.__dict__
