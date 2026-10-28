"""Prevent higher-order FLOP handlers from changing workload execution."""

from typing import Any

from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import TorchDispatchMode
from typing_extensions import override


class ExecutionGuard(TorchDispatchMode):
    """Reject opaque higher-order execution before PyTorch's FLOP mode sees it.

    Some PyTorch versions count Triton calls without executing their kernels and
    execute both branches of conditional operators. Neither is safe for sampling
    real training. Ordinary custom operators still execute through dispatch.
    """

    supports_higher_order_operators = True

    @override
    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Forward untyped dispatcher arguments or reject unsafe interception."""
        if isinstance(func, HigherOrderOperator):
            raise RuntimeError(
                "FLOP sampling does not support higher-order operators "
                f"({func}). Use eager ordinary operators for measured batches."
            )
        return func(*args, **(kwargs or {}))
