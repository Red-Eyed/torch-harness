"""Reliable utilities for PyTorch and Lightning."""

from importlib import import_module
from typing import TYPE_CHECKING, Literal

from torch_harness.flops import (
    FlopReport,
    InferenceProfile,
    UnmeasuredFlops,
    profile_inference,
    summarize_flops,
)
from torch_harness.layers import (
    SafeBatchNorm1d,
    SafeBatchNorm2d,
    SafeBatchNorm3d,
)
from torch_harness.losses import (
    BalancerConfig,
    LossBalancer,
    LossTarget,
    SuperLoss,
    UncertaintyWeighting,
    weighted_loss,
)
from torch_harness.runtime import replace_layers
from torch_harness.tensor_description import describe_tensor

if TYPE_CHECKING:
    from torch_harness.lightning import (
        FaultTolerantTrainingStep,
        FileModelStructure,
        FlopCounter,
        NonFiniteLossError,
    )
    from torch_harness.onnx import (
        collect_source_scopes,
        format_text_graph,
        write_text_graph,
    )

_OPTIONAL_EXPORTS: dict[str, Literal["lightning", "onnx"]] = {
    "FaultTolerantTrainingStep": "lightning",
    "FileModelStructure": "lightning",
    "FlopCounter": "lightning",
    "NonFiniteLossError": "lightning",
    "collect_source_scopes": "onnx",
    "format_text_graph": "onnx",
    "write_text_graph": "onnx",
}

__all__ = [
    "BalancerConfig",
    "FaultTolerantTrainingStep",
    "FileModelStructure",
    "FlopCounter",
    "FlopReport",
    "InferenceProfile",
    "LossBalancer",
    "LossTarget",
    "NonFiniteLossError",
    "SafeBatchNorm1d",
    "SafeBatchNorm2d",
    "SafeBatchNorm3d",
    "SuperLoss",
    "UncertaintyWeighting",
    "UnmeasuredFlops",
    "collect_source_scopes",
    "describe_tensor",
    "format_text_graph",
    "profile_inference",
    "replace_layers",
    "summarize_flops",
    "weighted_loss",
    "write_text_graph",
]


def __getattr__(name: str) -> object:
    """Load an optional public export on first access and cache its identity.

    Raises:
        AttributeError: If the name is not a public optional export.
        ModuleNotFoundError: If the required integration is not installed.
            Other import failures propagate without being relabeled.
    """
    integration = _OPTIONAL_EXPORTS.get(name)
    if integration is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import_module(integration)
        module = import_module(f".{integration}", __name__)
    except ModuleNotFoundError as error:
        if error.name != integration:
            raise
        raise ModuleNotFoundError(
            f"{name} requires the optional {integration} integration. "
            f"Install 'torch-harness[{integration}]'.",
            name=integration,
        ) from error
    value: object = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in discovery without importing their dependencies."""
    return sorted(set(globals()) | set(__all__))
