"""Measure real Lightning training batches with PyTorch operator formulas."""

from collections.abc import Callable, Mapping

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger
from typing_extensions import override

from torch_harness.flops import FlopReport, UnmeasuredFlops, summarize_flops
from torch_harness.flops._guard import ExecutionGuard


def _is_compiled(module: torch.nn.Module) -> bool:
    """Recognize torch.compile, Module.compile, and Lightning's unwrapped form."""
    return (
        hasattr(module, "_orig_mod")
        or getattr(module, "_compiled_call_impl", None) is not None
        or getattr(module, "_compiler_ctx", None) is not None
    )


class FlopCounter(Callback):
    """Count theoretical per-rank FLOPs without replaying model execution.

    Measures selected batches between this callback's batch-start and batch-end
    hooks. Backward hooks identify backward work, including recomputation, under
    both automatic and manual optimization. Other callbacks' hook ordering can
    affect included work. Counts exclude data loading, validation, communication,
    and unsupported operators; they are not hardware instruction measurements.

    Args:
        every_n_batches: Sampling interval within each epoch, starting at zero.
        max_batches: Maximum measured batches per fit, across all epochs.
        custom_mapping: Operator formulas following FlopCounterMode's contract.
            Keys are operator packets; functions receive tensor shapes and
            ``out_shape`` plus the operator's non-tensor arguments.
        log_metrics: Send batch and backward FLOPs to the Lightning logger.

    ``latest_report`` is UnmeasuredFlops until a sampled batch finishes. Reports
    are rank-local and are not checkpointed. Requires PyTorch's FlopCounterMode;
    compiled modules and higher-order operators are rejected. Opaque custom
    operators need formulas; unsupported ordinary operators contribute zero.
    """

    def __init__(
        self,
        *,
        every_n_batches: int = 1,
        max_batches: int = 1,
        custom_mapping: Mapping[object, Callable[..., int]] | None = None,
        log_metrics: bool = True,
    ) -> None:
        """Validate sampling configuration without activating dispatch modes."""
        super().__init__()
        if every_n_batches < 1 or max_batches < 1:
            raise ValueError("Sampling interval and maximum batches must be positive")
        self.every_n_batches = every_n_batches
        self.max_batches = max_batches
        self.custom_mapping = dict(custom_mapping or {})
        self.log_metrics = log_metrics
        self.latest_report: FlopReport | UnmeasuredFlops = UnmeasuredFlops()
        # Import lazily so older PyTorch versions can still use other callbacks.
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError as error:
            raise RuntimeError(
                "FlopCounter requires a PyTorch version providing FlopCounterMode"
            ) from error

        self._counter_type = FlopCounterMode
        self._counter: FlopCounterMode | None = None
        self._measured_batches = 0
        self._backward_start = 0
        self._backward_flops = 0
        self._guard: ExecutionGuard | None = None

    def _close(self) -> None:
        """Remove dispatch and module hooks, including after interrupted batches."""
        counter = self._counter
        self._counter = None
        guard = self._guard
        self._guard = None
        if guard is not None:
            guard.__exit__(None, None, None)
        if counter is not None:
            counter.__exit__(None, None, None)

    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset sampling and results when a callback is reused for another fit."""
        self._close()
        self._measured_batches = 0
        self.latest_report = UnmeasuredFlops()
        if any(_is_compiled(module) for module in pl_module.modules()):
            raise RuntimeError("FLOP sampling requires eager, uncompiled modules")

    @override
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Activate counting only for a selected real training batch."""
        self._close()
        if self._measured_batches >= self.max_batches:
            return
        if batch_idx % self.every_n_batches:
            return
        counter = self._counter_type(display=False, custom_mapping=self.custom_mapping)
        counter.__enter__()
        self._counter = counter
        guard = ExecutionGuard()
        guard.__enter__()
        self._guard = guard
        self._backward_flops = 0

    @override
    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor
    ) -> None:
        """Snapshot counts before each backward, including manual optimization."""
        if self._counter is not None:
            self._backward_start = self._counter.get_total_flops()

    @override
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Accumulate actual backward arithmetic rather than a forward multiplier."""
        if self._counter is not None:
            self._backward_flops += (
                self._counter.get_total_flops() - self._backward_start
            )

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Close measurement before building and optionally logging the report."""
        counter = self._counter
        if counter is None:
            return
        self._close()
        counts = {
            str(operator): count
            for operator, count in counter.get_flop_counts().get("Global", {}).items()
        }
        report = summarize_flops(counts, backward_flops=self._backward_flops)
        self.latest_report = report
        self._measured_batches += 1
        if trainer.is_global_zero:
            logger.info("Per-rank training batch FLOPs: {}", report)
        if self.log_metrics:
            pl_module.log_dict(
                {
                    "flops/batch": float(report.batch_flops),
                    "flops/backward": float(report.backward_flops),
                    "flops/non_backward": float(report.non_backward_flops),
                },
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
                rank_zero_only=True,
            )

    @override
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Discard partial measurements and preserve the original exception."""
        self._close()

    @override
    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Ensure dispatch hooks never outlive the fit lifecycle."""
        self._close()
