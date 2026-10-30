"""Measure native, eager stable, and compiled stable CE forward and backward.

Run with ``uv run benchmarks/stable_cross_entropy.py``. Uses CUDA when available,
otherwise CPU. Compilation and warmup are excluded from steady-state timings.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.utils.benchmark import Timer

from torch_harness import stable_cross_entropy, stable_distillation_cross_entropy
from torch_harness.functional import cross_entropy as compatible_cross_entropy

Objective = Callable[[Tensor, Tensor], Tensor]


def _native_hard(student: Tensor, targets: Tensor) -> Tensor:
    """Evaluate native CE with class-last inputs."""
    return F.cross_entropy(student, targets)


def _native_distillation(student: Tensor, teacher: Tensor) -> Tensor:
    """Evaluate ordinary soft-target CE with a detached teacher."""
    return -(teacher.softmax(-1) * student.log_softmax(-1)).sum(-1).mean()


def _forward_backward(objective: Objective, student: Tensor, target: Tensor) -> None:
    """Compute a fresh loss and student gradient without accumulating state."""
    torch.autograd.grad(objective(student, target), student)


def _check_compiled(
    objective: Objective, compiled: Objective, device: torch.device
) -> None:
    """Reject compiler paths that destroy the targeted small gradients."""
    if objective is stable_cross_entropy:
        for margin in (18.0, 40.0, 80.0):
            student = torch.tensor(
                [[margin, 0.0, 0.0]], device=device, requires_grad=True
            )
            target = torch.tensor([0], device=device)
            _compare_compiled(objective, compiled, student, target, atol=0.0)
        return
    teacher = torch.tensor([[2.0, 1.0, 0.0]], device=device)
    student = (teacher + teacher.new_tensor([[-1e-7, 1e-7, 0.0]])).requires_grad_()
    _compare_compiled(objective, compiled, student, teacher, atol=1e-14)
    identical = teacher.clone().requires_grad_()
    gradient = torch.autograd.grad(compiled(identical, teacher), identical)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))


def _compare_compiled(
    objective: Objective,
    compiled: Objective,
    student: Tensor,
    target: Tensor,
    atol: float,
) -> None:
    """Compare forward and backward without hiding tiny gradient regressions."""
    expected = objective(student, target)
    actual = compiled(student, target)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=0.0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student)[0],
        torch.autograd.grad(expected, student)[0],
        rtol=2e-5,
        atol=atol,
    )


def benchmark(include_compiled: bool = True) -> None:
    """Report FP32 timings at explicit small, medium, and vocabulary-sized shapes.

    CUDA timing uses the benchmark utility's synchronization. Peak allocated CUDA
    memory is measured after warmup, relative to the live input allocation.
    CPU timings use one thread. This is a synthetic microbenchmark, not a claim
    about end-to-end training throughput. Logs show progress for each variant.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "PyTorch {} device={} FP32; CPU timer threads=1", torch.__version__, device
    )
    objectives: tuple[tuple[str, Objective, Objective], ...] = (
        ("hard", _native_hard, stable_cross_entropy),
        ("distillation", _native_distillation, stable_distillation_cross_entropy),
    )
    for kind, native, stable in objectives:
        variants: list[tuple[str, Objective]] = [("native", native), ("stable", stable)]
        if kind == "hard":
            variants.append(("compatible stable", compatible_cross_entropy))
        if include_compiled:
            logger.info("Compiling and checking {}", kind)
            compiled = torch.compile(stable, fullgraph=True)
            try:
                _check_compiled(stable, compiled, device)
            except AssertionError:
                logger.opt(exception=True).warning(
                    "Rejecting compiled {}: numerical regression failed", kind
                )
            else:
                variants.append(("compiled stable", compiled))
        for rows, classes in ((128, 10), (256, 1000), (64, 32768)):
            generator = torch.Generator(device=device).manual_seed(0)
            student = torch.randn(
                rows, classes, device=device, generator=generator, requires_grad=True
            )
            if kind == "hard":
                target = torch.arange(rows, device=device) % classes
            else:
                target = torch.randn(rows, classes, device=device, generator=generator)
            for name, objective in variants:
                for backward in (False, True):
                    logger.info(
                        "Measuring {} {} shape={} backward={}",
                        kind,
                        name,
                        (rows, classes),
                        backward,
                    )
                    _measure(objective, student, target, backward)


def _measure(
    objective: Objective, student: Tensor, target: Tensor, backward: bool
) -> None:
    """Warm up and time one workload, optionally recording CUDA memory."""
    step = _forward_backward if backward else _forward_only
    step(objective, student, target)
    measurement = Timer(
        stmt="step(objective, student, target)",
        globals={
            "step": step,
            "objective": objective,
            "student": student,
            "target": target,
        },
    ).blocked_autorange(min_run_time=0.2)
    logger.info(
        "median={:.3f} ms IQR={:.3f} ms",
        measurement.median * 1000,
        measurement.iqr * 1000,
    )
    if student.device.type != "cuda":
        return
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step(objective, student, target)
    torch.cuda.synchronize()
    logger.info(
        "peak temporary allocation={:.2f} MiB",
        (torch.cuda.max_memory_allocated() - baseline) / 2**20,
    )


def _forward_only(objective: Objective, student: Tensor, target: Tensor) -> None:
    """Measure forward with autograd enabled, matching training forward costs."""
    objective(student, target)


if __name__ == "__main__":
    benchmark()
