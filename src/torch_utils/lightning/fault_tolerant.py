"""Fault-tolerant execution for Lightning training steps."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import wraps
from types import MethodType
from typing import Any, NamedTuple, TypeAlias

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from typing_extensions import override

_LOGGER = logger.bind(component="FaultTolerantTrainingStep")
_MISSING_ATTRIBUTE = object()

_TrainingStepTypeT: TypeAlias = Callable[..., STEP_OUTPUT]


class NonFiniteLossError(RuntimeError):
    """Indicate that ``training_step`` returned a NaN or infinite loss."""


class _FaultDecision(NamedTuple):
    """Describe whether a fault fits within the rolling error budget."""

    tolerated: bool
    faults_in_window: int


class _RollingFaultBudget:
    """Track a bounded number of faults in a monotonic rolling window."""

    def __init__(self, max_faults: int, window: timedelta) -> None:
        """Validate and store the rolling-window policy."""
        if isinstance(max_faults, bool) or max_faults < 1:
            msg = "max_faults must be a positive integer"
            raise ValueError(msg)
        if window <= timedelta(0):
            msg = "fault_window must be positive"
            raise ValueError(msg)

        self._max_faults = max_faults
        self._window_seconds = window.total_seconds()
        self._fault_times: deque[float] = deque()

    def record(self, occurred_at: float) -> _FaultDecision:
        """Record a fault when the configured rolling budget permits it."""
        cutoff = occurred_at - self._window_seconds
        while self._fault_times and self._fault_times[0] <= cutoff:
            self._fault_times.popleft()

        faults_in_window = len(self._fault_times) + 1
        tolerated = faults_in_window <= self._max_faults
        if tolerated:
            self._fault_times.append(occurred_at)

        return _FaultDecision(tolerated, faults_in_window)


class _NotInstalled(NamedTuple):
    """Represent a callback that has not wrapped a module."""


_NOT_INSTALLED = _NotInstalled()


class _TrainingStepInstallation(NamedTuple):
    """Retain the exact module state needed to undo method wrapping."""

    module: LightningModule
    original_instance_attribute: object


class FaultTolerantTrainingStep(Callback):
    """Skip infrequent training-step exceptions and non-finite losses.

    The callback wraps ``LightningModule.training_step`` during fitting. A tolerated
    fault returns ``None``, which makes Lightning automatic optimization skip backward
    and the optimizer update for that batch. Once more than ``max_faults`` occur in a
    rolling ``fault_window``, the active exception is re-raised with its traceback.

    This callback supports only automatic optimization on one process. Exceptions from
    backward or the optimizer update occur outside ``training_step`` and are therefore
    not recoverable by this callback.

    Example:
        Register a callback that skips up to three faults per rolling hour:

        >>> from datetime import timedelta
        >>> from lightning.pytorch import Trainer
        >>> from torch_utils.lightning import FaultTolerantTrainingStep
        >>> fault_tolerance = FaultTolerantTrainingStep(
        ...     max_faults=3,
        ...     fault_window=timedelta(hours=1),
        ... )
        >>> trainer = Trainer(callbacks=[fault_tolerance])

    Args:
        max_faults: Number of faults that may be skipped inside one rolling window.
        fault_window: Wall duration used to decide whether faults are too frequent.

    Raises:
        ValueError: If ``max_faults`` or ``fault_window`` is not positive.
        RuntimeError: During setup for manual or distributed optimization.
    """

    def __init__(self, *, max_faults: int, fault_window: timedelta) -> None:
        """Initialize the rolling fault budget."""
        self._max_faults = max_faults
        self._fault_window = fault_window
        self._budget = _RollingFaultBudget(max_faults, fault_window)
        self._installation: _TrainingStepInstallation | _NotInstalled = _NOT_INSTALLED

    @override
    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Wrap ``training_step`` when a supported fit begins."""
        if stage != "fit":
            return
        if not pl_module.automatic_optimization:
            msg = "FaultTolerantTrainingStep requires automatic optimization"
            raise RuntimeError(msg)
        if trainer.world_size != 1:
            msg = "FaultTolerantTrainingStep currently supports one process only"
            raise RuntimeError(msg)
        if not isinstance(self._installation, _NotInstalled):
            msg = "FaultTolerantTrainingStep is already installed"
            raise RuntimeError(msg)

        original_step = pl_module.training_step
        original_instance_attribute = pl_module.__dict__.get(
            "training_step", _MISSING_ATTRIBUTE
        )
        guarded_step = self._guard(original_step)
        pl_module.__dict__["training_step"] = MethodType(guarded_step, pl_module)
        self._installation = _TrainingStepInstallation(
            module=pl_module,
            original_instance_attribute=original_instance_attribute,
        )

    @override
    def teardown(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Restore the exact ``training_step`` attribute replaced during setup."""
        del trainer
        if stage != "fit" or isinstance(self._installation, _NotInstalled):
            return
        if self._installation.module is not pl_module:
            msg = "FaultTolerantTrainingStep was installed on a different module"
            raise RuntimeError(msg)

        original_attribute = self._installation.original_instance_attribute
        if original_attribute is _MISSING_ATTRIBUTE:
            del pl_module.__dict__["training_step"]
        else:
            pl_module.__dict__["training_step"] = original_attribute
        self._installation = _NOT_INSTALLED

    def _guard(self, training_step: _TrainingStepTypeT) -> Callable[..., STEP_OUTPUT]:
        """Build a bound-method-compatible wrapper around ``training_step``."""

        @wraps(training_step)
        def guarded_training_step(
            pl_module: LightningModule,
            *args: Any,  # noqa: ANN401 - Lightning training_step accepts arbitrary data.
            **kwargs: Any,  # noqa: ANN401 - Lightning training_step accepts arbitrary data.
        ) -> STEP_OUTPUT:
            del pl_module
            try:
                output = training_step(*args, **kwargs)
                if _has_non_finite_loss(output):
                    msg = "training_step returned a NaN or infinite loss"
                    raise NonFiniteLossError(msg)
            except Exception as exception:
                if isinstance(exception, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                if self._should_skip(exception, _batch_index(args, kwargs)):
                    return None
                raise
            return output

        return guarded_training_step

    def _should_skip(self, exception: Exception, batch_idx: object) -> bool:
        """Log a fault and decide whether its batch may be skipped."""
        decision = self._budget.record(time.monotonic())
        if decision.tolerated:
            _LOGGER.opt(exception=exception).warning(
                "Skipping training batch {} after fault {}/{} within {}: {}",
                batch_idx,
                decision.faults_in_window,
                self._max_faults,
                self._fault_window,
                exception,
            )
            return True

        _LOGGER.opt(exception=exception).error(
            "Fault tolerance exceeded at training batch {} with fault {}/{} "
            "within {}; propagating the exception",
            batch_idx,
            decision.faults_in_window,
            self._max_faults,
            self._fault_window,
        )
        return False


def _has_non_finite_loss(output: STEP_OUTPUT) -> bool:
    """Return whether a supported training-step output contains a bad loss."""
    if isinstance(output, torch.Tensor):
        loss = output
    elif isinstance(output, Mapping) and isinstance(output.get("loss"), torch.Tensor):
        loss = output["loss"]
    else:
        return False

    return not bool(torch.isfinite(loss.detach()).all().item())


def _batch_index(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    """Extract a batch index for diagnostics without constraining step signatures."""
    if "batch_idx" in kwargs:
        return kwargs["batch_idx"]
    if len(args) >= 2:
        return args[1]
    return "unknown"
