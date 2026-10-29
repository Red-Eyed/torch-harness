"""Shared eager operator counting and input metadata for profiling adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import torch

from torch_harness.flops.report import FlopReport, summarize_flops

if TYPE_CHECKING:
    from torch.utils.flop_counter import FlopCounterMode


def validate_eager(model: torch.nn.Module) -> None:
    """Reject compiled modules before counting can change their execution path."""
    for module in model.modules():
        if (
            hasattr(module, "_orig_mod")
            or getattr(module, "_compiled_call_impl", None) is not None
            or getattr(module, "_compiler_ctx", None) is not None
        ):
            raise RuntimeError("FLOP measurement requires eager, uncompiled modules")


def input_shapes(obj: object) -> Iterator[tuple[int, ...]]:
    """Extract tensor shapes from nested positional and keyword inputs."""
    match obj:
        case torch.Tensor():
            yield tuple(obj.shape)
        case Mapping():
            for value in cast(Mapping[object, object], obj).values():
                yield from input_shapes(value)
        case tuple() | list():
            for value in cast(tuple[object, ...] | list[object], obj):
                yield from input_shapes(value)


@contextmanager
def count_forward(
    custom_mapping: Mapping[object, Callable[..., int]],
) -> Iterator[FlopCounterMode]:
    """Activate operator counting with execution guards and exception cleanup."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as error:
        raise RuntimeError(
            "FLOP measurement requires a PyTorch version providing FlopCounterMode"
        ) from error
    from torch_harness.flops._guard import ExecutionGuard

    counter = FlopCounterMode(display=False, custom_mapping=dict(custom_mapping))
    with counter, ExecutionGuard():
        yield counter


def forward_report(
    counter: FlopCounterMode, shapes: tuple[tuple[int, ...], ...]
) -> FlopReport:
    """Convert global counts to an immutable report without module double counting."""
    counts = {
        str(operator): count
        for operator, count in counter.get_flop_counts().get("Global", {}).items()
    }
    return summarize_flops(counts, input_shapes=shapes)
