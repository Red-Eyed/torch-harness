"""Measure detached gradient statistics without accumulating parameter gradients."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Literal

import torch
import torch.distributed as distributed
from torch import Tensor, nn

GradientStatus = Literal["valid", "absent", "disconnected", "nonfinite", "tiny"]


@dataclass(frozen=True)
class GradientMeasurement:
    """Represent a usable norm or the explicit reason it cannot be used."""

    norm: float
    status: GradientStatus


def measurement_targets(shared: Tensor | Sequence[Tensor]) -> tuple[Tensor, ...]:
    """Validate a nonempty, unique target group on one device."""
    match shared:
        case Tensor():
            targets = (shared,)
        case _:
            targets = tuple(shared)
    if not targets or len({id(target) for target in targets}) != len(targets):
        raise ValueError("shared targets must be nonempty and unique")
    if any(
        not target.requires_grad or not target.is_floating_point() for target in targets
    ):
        raise ValueError("shared targets must require floating-point gradients")
    if any(target.device != targets[0].device for target in targets):
        raise ValueError("shared targets must use one device")
    return targets


def _tensor_norm(gradient: Tensor) -> float:
    """Compute a scaled norm with float32 or float64 tensor arithmetic."""
    values = gradient.detach()
    if values.is_sparse:
        values = values.coalesce().values()
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    working = values.to(dtype=dtype)
    if not working.numel():
        return 0.0
    scale = float(working.abs().max())
    if not isfinite(scale) or scale == 0:
        return scale
    # Scaling protects finite gradients whose squared values would overflow.
    return float(torch.linalg.vector_norm(working / scale)) * scale


def measure_gradient(loss: Tensor, targets: tuple[Tensor, ...]) -> GradientMeasurement:
    """Measure the concatenated L2 norm, preserving the graph for ordinary backward."""
    if not bool(loss.detach().isfinite()):
        return GradientMeasurement(0.0, "nonfinite")
    if not loss.requires_grad:
        return GradientMeasurement(0.0, "disconnected")
    gradients = torch.autograd.grad(
        loss, targets, retain_graph=True, create_graph=False, allow_unused=True
    )
    norms = [_tensor_norm(gradient) for gradient in gradients if gradient is not None]
    if not norms:
        return GradientMeasurement(0.0, "disconnected")
    norm = hypot(*norms)
    if not isfinite(norm) or any(not isfinite(value) for value in norms):
        return GradientMeasurement(0.0, "nonfinite")
    return GradientMeasurement(norm, "valid")


def measure_losses(
    losses: Mapping[str, Tensor],
    names: tuple[str, ...],
    targets: tuple[Tensor, ...],
    *,
    synchronize: bool,
    floor: float,
) -> dict[str, GradientMeasurement]:
    """Reduce mean rank norms and validity in one collective on scheduled updates.

    The statistic is the mean of rank-local norms, not the norm of the averaged
    parameter gradient. Any invalid or absent rank makes that loss unusable.
    Every rank must supply identical configuration and update steps.
    """
    active = synchronize and distributed.is_initialized()
    if active and any(isinstance(target, nn.Parameter) for target in targets):
        raise ValueError("DDP measurement requires shared activations, not parameters")
    measured = {
        name: measure_gradient(losses[name], targets)
        if name in losses
        else GradientMeasurement(0.0, "absent")
        for name in names
    }
    measured = {
        name: GradientMeasurement(sample.norm, "tiny")
        if sample.status == "valid" and sample.norm <= floor
        else sample
        for name, sample in measured.items()
    }
    if not active:
        return measured
    statuses: tuple[GradientStatus, ...] = (
        "valid",
        "absent",
        "disconnected",
        "nonfinite",
        "tiny",
    )
    rows = []
    for name in names:
        sample = measured[name]
        status = sample.status
        rows.append([sample.norm, *[float(status == item) for item in statuses]])
    packed = torch.tensor(rows, dtype=torch.float64, device=targets[0].device)
    world_size = distributed.get_world_size()
    packed[:, 0] /= world_size
    distributed.all_reduce(packed)
    output: dict[str, GradientMeasurement] = {}
    for name, row in zip(names, packed.cpu().tolist(), strict=True):
        status: GradientStatus = "valid"
        for index, item in enumerate(statuses[1:], 2):
            if row[index]:
                status = item
                break
        output[name] = GradientMeasurement(row[0], status)
    return output
