"""Print finite-value statistics for tensors in nested structures."""

from collections.abc import Iterable, Mapping
from typing import cast

import torch

_StatisticValue = str | int | float | tuple[int, ...]


def _tensor_rows(
    tensor: torch.Tensor, percentiles: tuple[float, ...]
) -> list[tuple[str, _StatisticValue]]:
    """Calculate display rows without modifying the tensor or recording gradients."""
    if tensor.is_complex():
        raise TypeError("describe_tensor does not support complex tensors")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise TypeError("describe_tensor requires dense, non-quantized tensors")
    if tensor.device.type == "meta":
        raise ValueError("describe_tensor requires materialized tensor values")

    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    values = tensor.detach().flatten().to(dtype=dtype)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    rows: list[tuple[str, _StatisticValue]] = [
        ("shape", tuple(tensor.shape)),
        ("dtype", str(tensor.dtype)),
        ("device", str(tensor.device)),
        ("count", values.numel()),
        ("null_count", values.numel() - finite.numel()),
    ]
    if not finite.numel():
        return rows

    std, mean = torch.std_mean(finite, unbiased=False)
    rows.extend(
        [("mean", mean.item()), ("std", std.item()), ("min", finite.min().item())]
    )
    if percentiles:
        quantiles = torch.quantile(finite, finite.new_tensor(percentiles))
        rows.extend(
            (f"p{percentile * 100:g}", value.item())
            for percentile, value in zip(percentiles, quantiles, strict=True)
        )
    rows.append(("max", finite.max().item()))
    return rows


def _describe(obj: object, percentiles: tuple[float, ...], prefix: str) -> None:
    """Traverse containers and print each tensor's statistics in encounter order."""
    match obj:
        case torch.Tensor():
            rows = _tensor_rows(obj, percentiles)
            print(f"\n{prefix}")
            for key, value in rows:
                print(f"  {key:<12} {value}")
        case Mapping():
            for key, value in cast(Mapping[object, object], obj).items():
                _describe(value, percentiles, f"{prefix}.{key}")
        case str() | bytes():
            return
        case Iterable():
            for index, value in enumerate(cast(Iterable[object], obj)):
                _describe(value, percentiles, f"{prefix}[{index}]")


def describe_tensor(
    obj: object,
    *,
    percentiles: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.99),
    prefix: str = "root",
) -> None:
    """Print tensor statistics recursively to standard output.

    Mapping paths use keys and iterable paths use indexes. Non-tensor leaves,
    strings, and bytes are ignored. Containers must be finite and acyclic;
    iterators are consumed. Inputs and gradients are not modified.

    Statistics exclude NaN and infinities; ``null_count`` counts all such values.
    Empty and entirely non-finite tensors print metadata only. Float64 precision
    is retained; other real numeric dtypes use float32 for calculations, which
    can round large integers. Standard deviation uses the population convention.

    Quantiles use all finite values without sampling. Calculation allocates
    temporary tensors, may synchronize accelerators, and follows PyTorch's
    quantile device and size limits. Output before a later failure is retained.

    Args:
        obj: A tensor or nested mappings and iterables containing tensors.
        percentiles: Quantile fractions in [0, 1], printed in supplied order.
            An empty tuple omits percentile rows.
        prefix: Path printed for the root object.

    Raises:
        ValueError: If a percentile is non-finite or outside [0, 1], or a tensor
            has no materialized values.
        TypeError: If a tensor is complex, non-strided, or quantized.
    """
    if any(not 0 <= percentile <= 1 for percentile in percentiles):
        raise ValueError("percentiles must be finite fractions between 0 and 1")
    _describe(obj, percentiles, prefix)
