"""Forward and backward numerical contracts for stable cross-entropy."""

from collections.abc import Callable
from importlib import import_module
from math import exp, log1p
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

import torch_harness as th
from torch_harness.losses.stable_cross_entropy.cross_entropy import Reduction

Normalize = Callable[[Tensor, int], Tensor]


@pytest.mark.parametrize("margin", [18.0, 40.0, 80.0])
def test_confident_hard_target_retains_forward_and_backward(margin: float) -> None:
    """Retain tiny losses and all gradient entries past the FP32 cliff."""
    logits = torch.tensor([margin, 0.0, 0.0], requires_grad=True)
    loss = th.stable_cross_entropy(logits, torch.tensor(0))
    gradient = torch.autograd.grad(loss, logits)[0]
    other = exp(-margin) / (1.0 + 2.0 * exp(-margin))
    torch.testing.assert_close(
        loss.double(),
        torch.tensor(log1p(2.0 * exp(-margin)), dtype=torch.float64),
        rtol=2e-6,
        atol=0.0,
    )
    torch.testing.assert_close(
        gradient.double(),
        torch.tensor([-2 * other, other, other], dtype=torch.float64),
        rtol=2e-6,
        atol=0.0,
    )
    assert gradient[0] < 0
    assert gradient.sum() == 0


@pytest.mark.parametrize("normalize", [th.stable_log_softmax, th.stable_softmax])
def test_confident_normalization_backward(normalize: Normalize) -> None:
    """Preserve the maximum's derivative even when its probability rounds to one."""
    logits = torch.tensor([9.0, -9.0, -9.0], requires_grad=True)
    actual = normalize(logits, -1)
    gradient = torch.autograd.grad(actual[0], logits)[0]
    reference = logits.detach().double().requires_grad_()
    expected = reference.log_softmax(-1)
    if normalize is th.stable_softmax:
        expected = expected.exp()
    expected_gradient = torch.autograd.grad(expected[0], reference)[0]
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-14)
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=1e-6, atol=1e-14
    )
    assert gradient[0] > 0


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("normalize", [th.stable_log_softmax, th.stable_softmax])
def test_normalization_matches_native_with_nonuniform_upstream(
    dim: int, normalize: Normalize
) -> None:
    """Match native forward and backward for ties, strides, and varied axes."""
    logits = torch.tensor(
        [[2.0, 2.0, -4.0], [-3.0, 0.5, 1.0]], dtype=torch.float64
    ).T.requires_grad_()
    before = logits.detach().clone()
    upstream = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    actual = normalize(logits, dim)
    expected = logits.log_softmax(dim)
    if normalize is th.stable_softmax:
        expected = logits.softmax(dim)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, upstream)[0],
        torch.autograd.grad(expected, logits, upstream)[0],
    )
    torch.testing.assert_close(logits, before)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_hard_target_matches_native_forward_backward(reduction: Reduction) -> None:
    """Match native spatial CE for misclassification and nonuniform gradients."""
    logits = torch.linspace(-3, 3, 24, dtype=torch.float64).reshape(2, 3, 4)
    logits = logits.requires_grad_()
    targets = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    actual = th.stable_cross_entropy(logits, targets, dim=1, reduction=reduction)
    expected = F.cross_entropy(logits, targets, reduction=reduction)
    upstream = torch.arange(1, 9, dtype=torch.float64).reshape(2, 4)
    if reduction != "none":
        upstream = torch.tensor(2.5, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, upstream)[0],
        torch.autograd.grad(expected, logits, upstream)[0],
    )


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_distillation_matches_native_forward_backward(reduction: Reduction) -> None:
    """Match CE and upstream gradient scaling away from cancellation."""
    student = torch.tensor(
        [[1.0, -3.0, 2.0], [0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True
    )
    teacher = torch.tensor(
        [[-1.0, 3.0, 2.0], [0.2, 0.1, 0.3]], dtype=torch.float64, requires_grad=True
    )
    actual = th.stable_distillation_cross_entropy(student, teacher, reduction=reduction)
    expected = -(teacher.detach().softmax(-1) * student.log_softmax(-1)).sum(-1)
    if reduction == "mean":
        expected = expected.mean()
    if reduction == "sum":
        expected = expected.sum()
    upstream = torch.tensor([0.5, 2.0], dtype=torch.float64)
    if reduction != "none":
        upstream = torch.tensor(2.5, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student, upstream, retain_graph=True)[0],
        torch.autograd.grad(expected, student, upstream)[0],
    )
    assert torch.autograd.grad(actual.sum(), teacher, allow_unused=True)[0] is None


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_identical_distillation_has_zero_gradient(dtype: torch.dtype) -> None:
    """Equal logits yield teacher entropy and an exactly zero student gradient."""
    student = torch.tensor(
        [[2.0, 1.0, 0.0], [40.0, 0.0, -40.0]], dtype=dtype, requires_grad=True
    )
    teacher = student.detach().clone().requires_grad_()
    loss = th.stable_distillation_cross_entropy(student, teacher, reduction="sum")
    gradient = torch.autograd.grad(loss, student)[0]
    assert torch.equal(gradient, torch.zeros_like(student))
    reference = teacher.detach().double()
    entropy = -(reference.softmax(-1) * reference.log_softmax(-1)).sum()
    torch.testing.assert_close(loss.double(), entropy, rtol=1e-6, atol=1e-14)


def test_near_equal_distillation_matches_float64_gradient() -> None:
    """Resolve the small FP32 probability differences in the paper's example."""
    teacher = torch.tensor([2.0, 1.0, 0.0])
    student = (teacher + torch.tensor([-1e-7, 1e-7, 0.0])).requires_grad_()
    loss = th.stable_distillation_cross_entropy(student, teacher)
    gradient = torch.autograd.grad(loss, student)[0]
    expected = student.detach().double().softmax(-1) - teacher.double().softmax(-1)
    torch.testing.assert_close(gradient.double(), expected, rtol=2e-6, atol=1e-15)
    assert gradient.sum().abs() < 1e-14


@pytest.mark.parametrize("offset", [1.0, 1000.0])
def test_distillation_is_invariant_to_constant_logit_shift(offset: float) -> None:
    """Equivalent teacher and student distributions have zero gradients."""
    teacher = torch.tensor([2.0, 1.0, 0.0])
    student = (teacher + offset).requires_grad_()
    loss = th.stable_distillation_cross_entropy(student, teacher)
    assert torch.equal(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_distant_distillation_avoids_overflow_in_unused_branch() -> None:
    """Large opposing logits retain finite forward values and gradients."""
    student = torch.tensor([1000.0, -1000.0, 0.0], requires_grad=True)
    teacher = -student.detach()
    loss = th.stable_distillation_cross_entropy(student, teacher)
    gradient = torch.autograd.grad(loss, student)[0]
    torch.testing.assert_close(loss, torch.tensor(2000.0))
    torch.testing.assert_close(gradient, torch.tensor([1.0, -1.0, 0.0]))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_accumulation_dtype_and_input_gradient_dtype(dtype: torch.dtype) -> None:
    """Promote low precision outputs while preserving input gradient dtype."""
    logits = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    outputs = (
        th.stable_log_softmax(logits),
        th.stable_softmax(logits),
        th.stable_cross_entropy(logits, torch.tensor(1)),
        th.stable_distillation_cross_entropy(logits, -logits.detach()),
    )
    for output in outputs:
        assert output.dtype == expected_dtype
        gradient = torch.autograd.grad(output.sum(), logits, retain_graph=True)[0]
        assert gradient.dtype == dtype
        assert bool(gradient.isfinite().all())


def test_single_class_and_empty_batch() -> None:
    """Handle singleton classes and standard empty reduction semantics."""
    logits = torch.tensor([[3.0]], requires_grad=True)
    loss = th.stable_cross_entropy(logits, torch.tensor([0]))
    assert loss == 0
    assert torch.autograd.grad(loss, logits)[0].item() == 0
    empty = torch.empty(0, 3, requires_grad=True)
    targets = torch.empty(0, dtype=torch.int64)
    assert th.stable_cross_entropy(empty, targets, reduction="none").shape == (0,)
    assert th.stable_cross_entropy(empty, targets, reduction="sum") == 0
    assert th.stable_cross_entropy(empty, targets).isnan()


def test_hard_target_gradcheck_and_gradgradcheck() -> None:
    """Validate first and second derivatives of the ordinary autograd path."""
    logits = torch.tensor([[0.2, -0.4, 1.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1])

    def objective(values: Tensor) -> Tensor:
        """Evaluate the hard-target loss at fixed labels."""
        return th.stable_cross_entropy(values, targets)

    assert torch.autograd.gradcheck(objective, (logits,))
    assert torch.autograd.gradgradcheck(objective, (logits,))


def test_distillation_gradcheck_and_first_order_contract() -> None:
    """Check the substituted derivative and expose its detached nature."""
    student = torch.tensor([0.2, -0.4, 1.0], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64)

    def objective(values: Tensor) -> Tensor:
        """Evaluate distillation against a fixed teacher."""
        return th.stable_distillation_cross_entropy(values, teacher)

    assert torch.autograd.gradcheck(objective, (student,))
    gradient = torch.autograd.grad(objective(student), student, create_graph=True)[0]
    assert not gradient.requires_grad


@pytest.mark.parametrize(
    ("logits", "dim", "error"),
    [
        (torch.tensor([1, 2]), -1, TypeError),
        (torch.tensor([1j]), -1, TypeError),
        (torch.tensor(1.0), -1, ValueError),
        (torch.ones(2, 3), 2, ValueError),
        (torch.empty(2, 0), -1, ValueError),
    ],
)
def test_invalid_normalization_inputs(
    logits: Tensor, dim: int, error: type[Exception]
) -> None:
    """Reject unsupported dtypes and nonexistent class axes."""
    with pytest.raises(error):
        th.stable_log_softmax(logits, dim)


def test_invalid_loss_inputs() -> None:
    """Reject malformed labels, teacher tensors, and reductions."""
    logits = torch.ones(2, 3)
    with pytest.raises(TypeError, match="int64"):
        th.stable_cross_entropy(logits, torch.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        th.stable_cross_entropy(logits, torch.zeros(2, 1, dtype=torch.int64))
    with pytest.raises(RuntimeError):
        th.stable_cross_entropy(logits, torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="reduction"):
        th.stable_cross_entropy(
            logits, torch.tensor([0, 1]), reduction=cast(Reduction, "bad")
        )
    with pytest.raises(ValueError, match="shape"):
        th.stable_distillation_cross_entropy(logits, torch.ones(3))
    with pytest.raises(TypeError, match="dtype"):
        th.stable_distillation_cross_entropy(logits, logits.double())


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_distillation_axes_and_noncontiguous_inputs(dim: int) -> None:
    """Match teacher CE and conserved gradients along any class axis."""
    student = torch.tensor(
        [[-0.5, 0.2, 1.0], [0.1, -0.3, 0.2]], dtype=torch.float64
    ).T.requires_grad_()
    teacher = -student.detach()
    actual = th.stable_distillation_cross_entropy(student, teacher, dim=dim)
    expected = -(teacher.softmax(dim) * student.log_softmax(dim)).sum(dim).mean()
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, student)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, student)[0])
    torch.testing.assert_close(gradient.sum(dim), torch.zeros_like(gradient.sum(dim)))


@pytest.mark.parametrize("kind", ["hard", "distillation", "normalization"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_forward_backward(kind: str) -> None:
    """Compare CUDA eager kernels with FP64 CPU forward and gradient references."""
    student = torch.tensor(
        [[9.0, -9.0, -9.0], [0.2, 0.4, -0.1]], device="cuda", requires_grad=True
    )
    reference = student.detach().cpu().double().requires_grad_()
    if kind == "hard":
        targets = torch.tensor([0, 2], device="cuda")
        actual = th.stable_cross_entropy(student, targets)
        expected = th.stable_cross_entropy(reference, targets.cpu())
    elif kind == "distillation":
        teacher = student.detach() + student.new_tensor(
            [[0.0, 0.0, 0.0], [1e-6, 0.0, -1e-6]]
        )
        actual = th.stable_distillation_cross_entropy(student, teacher)
        expected = th.stable_distillation_cross_entropy(
            reference, teacher.cpu().double()
        )
    else:
        actual = th.stable_log_softmax(student)[:, 0].sum()
        expected = th.stable_log_softmax(reference)[:, 0].sum()
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-6, atol=1e-14)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student)[0].cpu().double(),
        torch.autograd.grad(expected, reference)[0],
        rtol=2e-5,
        atol=1e-14,
    )


def test_component_and_namespace_exports() -> None:
    """Expose identical functions through all supported import paths."""
    import torch_harness.losses as losses

    component = import_module("torch_harness.losses.stable_cross_entropy")
    for name in component.__all__:
        assert getattr(th, name) is getattr(losses, name)
        assert getattr(th, name) is getattr(component, name)
