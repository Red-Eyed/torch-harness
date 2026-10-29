"""Measure a single real model forward for paper-oriented FLOP reporting."""

from collections.abc import Callable, Iterator, Mapping
from typing import cast

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger
from torch.utils.hooks import RemovableHandle
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


def _input_shapes(obj: object) -> Iterator[tuple[int, ...]]:
    """Extract tensor shapes from nested positional and keyword inputs."""
    match obj:
        case torch.Tensor():
            yield tuple(obj.shape)
        case Mapping():
            for value in cast(Mapping[object, object], obj).values():
                yield from _input_shapes(value)
        case tuple() | list():
            for value in cast(tuple[object, ...] | list[object], obj):
                yield from _input_shapes(value)


class FlopCounter(Callback):
    """Report theoretical FLOPs for the first model forward in the first batch.

    Hooks observe an existing call without replaying inputs or changing model
    mode. Counting stops when forward returns, before loss, backward, or optimizer
    work. Subsequent forwards, batches, and epochs are never measured. State
    resets per fit; reports are not checkpointed. Each rank measures locally.

    Args:
        module: Model whose forward to observe. Defaults to the LightningModule;
            training_step must call it through self(...). Select an inner model
            when training_step calls that model directly instead.
        custom_mapping: PyTorch operator packets mapped to shape-based formulas,
            following FlopCounterMode's custom_mapping contract.
        log_metrics: Send the single forward FLOP metric to Lightning's logger.

    Input metadata includes tensor shapes nested in tuples, lists, and mappings.
    Batch size assumes the first non-scalar input is batch-first. Counts cover
    the full observed input, with one multiply-add equal to two FLOPs. Unsupported
    ordinary operators contribute zero. Compiled modules and higher-order
    operators are rejected. See the component documentation before citing counts.
    """

    def __init__(
        self,
        *,
        module: torch.nn.Module | None = None,
        custom_mapping: Mapping[object, Callable[..., int]] | None = None,
        log_metrics: bool = True,
    ) -> None:
        """Store configuration without installing hooks or activating counting."""
        super().__init__()
        self._module = module
        self.custom_mapping = dict(custom_mapping or {})
        self.log_metrics = log_metrics
        self.latest_report: FlopReport | UnmeasuredFlops = UnmeasuredFlops()
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError as error:
            raise RuntimeError(
                "FlopCounter requires a PyTorch version providing FlopCounterMode"
            ) from error
        self._counter_type = FlopCounterMode
        self._counter: FlopCounterMode | None = None
        self._guard: ExecutionGuard | None = None
        self._hooks: list[RemovableHandle] = []
        self._attempted = False
        self._pending_log = False
        self._forward_depth = 0
        self._shapes: tuple[tuple[int, ...], ...] = ()

    def _close(self) -> None:
        """Remove measurement hooks and dispatch contexts, including after failure."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        guard = self._guard
        self._guard = None
        if guard is not None:
            guard.__exit__(None, None, None)
        counter = self._counter
        self._counter = None
        if counter is not None:
            counter.__exit__(None, None, None)
        self._forward_depth = 0

    def _before_forward(
        self,
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Start counting at the outermost invocation of the selected model."""
        self._forward_depth += 1
        if self._forward_depth != 1:
            return
        self._shapes = tuple(_input_shapes((args, kwargs)))
        counter = self._counter_type(display=False, custom_mapping=self.custom_mapping)
        counter.__enter__()
        self._counter = counter
        guard = ExecutionGuard()
        guard.__enter__()
        self._guard = guard

    def _after_forward(
        self,
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> None:
        """Stop before returning model outputs to the training step."""
        self._forward_depth -= 1
        if self._forward_depth:
            return
        counter = self._counter
        if counter is None:
            return
        self._close()
        counts = {
            str(operator): count
            for operator, count in counter.get_flop_counts().get("Global", {}).items()
        }
        self.latest_report = summarize_flops(counts, input_shapes=self._shapes)
        self._pending_log = True

    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset the one-shot lifecycle and validate eager execution."""
        self._close()
        self._attempted = False
        self._pending_log = False
        self.latest_report = UnmeasuredFlops()
        model = self._module if self._module is not None else pl_module
        if any(_is_compiled(module) for module in model.modules()):
            raise RuntimeError("FLOP measurement requires eager, uncompiled modules")

    @override
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Arm hooks once, excluding validation and sanity-check forwards."""
        if self._attempted:
            return
        self._attempted = True
        model = self._module if self._module is not None else pl_module
        self._hooks.append(
            model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        )
        self._hooks.append(
            model.register_forward_hook(
                self._after_forward, with_kwargs=True, prepend=True
            )
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
        """Publish once after closing any incomplete first-forward measurement."""
        missed_forward = bool(self._hooks)
        self._close()
        if missed_forward and trainer.is_global_zero:
            logger.warning(
                "No complete model forward observed in the first training batch. "
                "Call self(...) or set FlopCounter(module=your_model)."
            )
        if not self._pending_log:
            return
        self._pending_log = False
        report = self.latest_report
        if not isinstance(report, FlopReport):
            return
        if trainer.is_global_zero:
            logger.info("First model forward FLOPs (multiply-add = 2): {}", report)
        if self.log_metrics and trainer.is_global_zero:
            for metric_logger in trainer.loggers:
                metric_logger.log_metrics(
                    {"flops/forward": float(report.forward_flops)},
                    step=trainer.global_step,
                )

    @override
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Clean up without replacing the exception or publishing partial counts."""
        self._close()
        self._pending_log = False

    @override
    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Ensure dispatch and forward hooks never outlive the fit lifecycle."""
        self._close()
