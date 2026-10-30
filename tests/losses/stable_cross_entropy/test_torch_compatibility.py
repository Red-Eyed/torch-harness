"""Native PyTorch API parity and stable forward/backward regressions."""

import inspect
from collections.abc import Callable
from io import BytesIO
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torch_harness as th
import torch_harness.functional as HF
from torch_harness.losses.stable_cross_entropy.cross_entropy import Reduction


@pytest.fixture(params=[(3,), (2, 3), (2, 3, 2, 2)])
def logits(request: pytest.FixtureRequest) -> Tensor:
    """Build deterministic unbatched, batch, and spatial logits."""
    shape = cast(tuple[int, ...], request.param)
    count = torch.Size(shape).numel()
    return (
        torch.linspace(-1, 1, count, dtype=torch.float64)
        .reshape(shape)
        .requires_grad_()
    )


def _targets(logits: Tensor) -> Tensor:
    """Build native-layout class indices for a known logits tensor."""
    shape = logits.shape[:1] + logits.shape[2:] if logits.ndim > 1 else torch.Size([])
    return (torch.arange(shape.numel()) % 3).reshape(shape)


def _assert_forward_backward(
    actual: Tensor, expected: Tensor, inputs: tuple[Tensor, ...]
) -> None:
    """Compare values and all requested derivatives under nonuniform upstreams."""
    torch.testing.assert_close(actual, expected, equal_nan=True)
    upstream = torch.linspace(0.5, 2.0, actual.numel(), dtype=actual.dtype).reshape(
        actual.shape
    )
    actual_gradients = torch.autograd.grad(actual, inputs, upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(
        expected, inputs, upstream, retain_graph=True
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, equal_nan=True)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ignored", [False, True])
def test_index_forward_backward_parity(
    logits: Tensor,
    reduction: Reduction,
    smoothing: float,
    weighted: bool,
    ignored: bool,
) -> None:
    """Match native weights, smoothing, ignored targets, and mean denominators."""
    targets = _targets(logits)
    if ignored:
        targets = targets.clone()
        targets.reshape(-1)[0] = -7
    weight = torch.tensor([0.5, 2.0, 0.0], dtype=logits.dtype) if weighted else None
    actual = HF.cross_entropy(
        logits,
        targets,
        weight,
        ignore_index=-7,
        reduction=reduction,
        label_smoothing=smoothing,
    )
    expected = F.cross_entropy(
        logits,
        targets,
        weight,
        ignore_index=-7,
        reduction=reduction,
        label_smoothing=smoothing,
    )
    _assert_forward_backward(actual, expected, (logits,))


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_probability_forward_backward_parity(
    logits: Tensor, reduction: Reduction, smoothing: float, weighted: bool
) -> None:
    """Preserve derivatives to logits, probability targets, and class weights."""
    dim = 0 if logits.ndim == 1 else 1
    targets = (-logits.detach()).softmax(dim).requires_grad_()
    weight = (
        torch.tensor([0.5, 2.0, 0.0], dtype=logits.dtype, requires_grad=True)
        if weighted
        else None
    )
    inputs = (logits, targets) if weight is None else (logits, targets, weight)
    actual = HF.cross_entropy(
        logits, targets, weight, reduction=reduction, label_smoothing=smoothing
    )
    expected = F.cross_entropy(
        logits, targets, weight, reduction=reduction, label_smoothing=smoothing
    )
    _assert_forward_backward(actual, expected, inputs)


@pytest.mark.parametrize("operation", ["softmax", "log_softmax"])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_normalization_noncontiguous_forward_backward(operation: str, dim: int) -> None:
    """Match native normalization at explicit axes with noncontiguous storage."""
    input = torch.tensor(
        [[2.0, 2.0, 0.1], [-0.5, 0.2, 1.0]], dtype=torch.float64
    ).T.requires_grad_()
    actual = getattr(HF, operation)(input, dim=dim)
    expected = getattr(F, operation)(input, dim=dim)
    _assert_forward_backward(actual, expected, (input,))


@pytest.mark.parametrize("operation", ["softmax", "log_softmax"])
@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (2, 3, 4), (2, 3, 2, 2)])
def test_implicit_normalization_dimension(
    operation: str, shape: tuple[int, ...]
) -> None:
    """Match deprecated dimension inference and warning behavior for each rank."""
    input = (
        torch.arange(torch.Size(shape).numel(), dtype=torch.float64)
        .reshape(shape)
        .requires_grad_()
    )
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        actual = getattr(HF, operation)(input)
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        expected = getattr(F, operation)(input)
    _assert_forward_backward(actual, expected, (input,))


@pytest.mark.parametrize("operation", ["softmax", "log_softmax"])
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
def test_empty_normalization(operation: str, shape: tuple[int, ...]) -> None:
    """Preserve empty shapes and backward support."""
    input = torch.empty(shape, dtype=torch.float64, requires_grad=True)
    actual = getattr(HF, operation)(input, dim=-1)
    expected = getattr(F, operation)(input, dim=-1)
    _assert_forward_backward(actual, expected, (input,))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("autocast", [False, True])
def test_dtype_and_cpu_autocast_parity(dtype: torch.dtype, autocast: bool) -> None:
    """Preserve native dtype and gradient dtype under and outside autocast."""
    input = torch.tensor([[0.2, -0.4, 1.0]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([2])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = HF.cross_entropy(input, targets)
        expected = F.cross_entropy(input, targets)
        for operation in ("softmax", "log_softmax"):
            normalized = getattr(HF, operation)(input, dim=1)
            native = getattr(F, operation)(input, dim=1)
            assert normalized.dtype == native.dtype
            torch.testing.assert_close(normalized, native, rtol=0.02, atol=0.01)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    torch.testing.assert_close(
        torch.autograd.grad(actual, input)[0],
        torch.autograd.grad(expected, input)[0],
        rtol=0.02,
        atol=0.01,
    )


@pytest.mark.parametrize("operation", ["softmax", "log_softmax"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_explicit_normalization_dtype(operation: str, dtype: torch.dtype) -> None:
    """Honor explicit dtype casting even when the source contains integers."""
    input = torch.tensor([[1, 2, 3]])
    actual = getattr(HF, operation)(input, 1, dtype=dtype)
    expected = getattr(F, operation)(input, 1, dtype=dtype)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("size_average", [None, False, True])
@pytest.mark.parametrize("reduce", [None, False, True])
def test_legacy_reduction_precedence(
    size_average: bool | None, reduce: bool | None
) -> None:
    """Deprecated positional flags override a conflicting reduction consistently."""
    input = torch.tensor([[0.1, 0.2, 0.3], [1.0, -0.1, 0.5]], requires_grad=True)
    targets = torch.tensor([1, 2])
    if size_average is None and reduce is None:
        return
    with pytest.warns(UserWarning, match="size_average and reduce"):
        actual = HF.cross_entropy(
            input, targets, None, size_average, -100, reduce, "none", 0.1
        )
    with pytest.warns(UserWarning, match="size_average and reduce"):
        expected = F.cross_entropy(
            input, targets, None, size_average, -100, reduce, "none", 0.1
        )
    _assert_forward_backward(actual, expected, (input,))


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2])
def test_all_ignored_and_empty_batches(reduction: Reduction, smoothing: float) -> None:
    """Match NaN means, zero sums, and zero gradients when every item is absent."""
    for count in (0, 2):
        input = torch.zeros(count, 3, dtype=torch.float64, requires_grad=True)
        targets = torch.full((count,), -100)
        actual = HF.cross_entropy(
            input, targets, reduction=reduction, label_smoothing=smoothing
        )
        expected = F.cross_entropy(
            input, targets, reduction=reduction, label_smoothing=smoothing
        )
        _assert_forward_backward(actual, expected, (input,))


def test_confident_loss_and_normalization_gradients() -> None:
    """Keep the numerical fix through the compatible functional and module APIs."""
    input = torch.tensor([[9.0, -9.0, -9.0]], requires_grad=True)
    targets = torch.tensor([0])
    expected = th.stable_cross_entropy(input, targets)
    for actual in (
        HF.cross_entropy(input, targets),
        th.StableCrossEntropyLoss()(input, targets),
    ):
        _assert_forward_backward(actual, expected, (input,))
        assert actual > 0
    for operation in ("softmax", "log_softmax"):
        actual = getattr(HF, operation)(input, dim=1)[0, 0]
        gradient = torch.autograd.grad(actual, input, retain_graph=True)[0]
        assert gradient[0, 0] > 0
        torch.testing.assert_close(
            gradient.sum(), torch.tensor(0.0), atol=1e-14, rtol=0
        )


def test_module_native_state_and_lifecycle() -> None:
    """Retain constructor, weight buffers, serialization, and native type identity."""
    weight = torch.tensor([0.5, 2.0, 1.0])
    native = nn.CrossEntropyLoss(
        weight, ignore_index=-7, label_smoothing=0.2, reduction="sum"
    )
    stable = th.StableCrossEntropyLoss(
        weight.clone(), ignore_index=-7, label_smoothing=0.2, reduction="sum"
    )
    assert isinstance(stable, nn.CrossEntropyLoss)
    assert dict(stable.named_buffers()).keys() == {"weight"}
    assert list(stable.parameters()) == []
    buffer = BytesIO()
    torch.save(native.state_dict(), buffer)
    buffer.seek(0)
    stable.load_state_dict(torch.load(buffer, weights_only=True))
    stable = stable.double()
    native = native.double()
    input = torch.tensor(
        [[0.2, -0.1, 1.0], [-0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([0, -7])
    _assert_forward_backward(stable(input, target), native(input, target), (input,))
    assert stable.weight is not None
    assert stable.weight.dtype == torch.float64


def test_module_real_optimization_step() -> None:
    """Use the replacement in a real module, backward, and optimizer lifecycle."""
    model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]]))
    criterion = th.StableCrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input = torch.tensor([[1.0, 0.5], [-0.5, 1.0]])
    targets = torch.tensor([0, 2])
    before = criterion(model(input), targets)
    before.backward()
    optimizer.step()
    assert criterion(model(input), targets) < before.detach()


@pytest.mark.parametrize("name", ["softmax", "log_softmax", "cross_entropy"])
def test_function_signature_parity(name: str) -> None:
    """Match parameter order, defaults, and calling convention of native functions."""
    actual = inspect.signature(getattr(HF, name)).parameters
    expected = inspect.signature(getattr(F, name)).parameters
    assert list(actual) == list(expected)
    for key in expected:
        assert actual[key].default == expected[key].default
        assert actual[key].kind == expected[key].kind


@pytest.mark.parametrize("operation", [HF.softmax, HF.log_softmax])
def test_normalization_errors(operation: Callable[..., Tensor]) -> None:
    """Keep native invalid-axis and unsupported-dtype errors."""
    with pytest.raises(IndexError):
        operation(torch.ones(2, 3), dim=2)
    with pytest.raises(RuntimeError):
        operation(torch.ones(2, 3, dtype=torch.int64), dim=1)


def test_cross_entropy_errors() -> None:
    """Reject malformed targets, weights, smoothing, and reduction values."""
    input = torch.ones(2, 3)
    target = torch.tensor([0, 1])
    with pytest.raises(IndexError):
        HF.cross_entropy(input, torch.tensor([0, 3]))
    with pytest.raises(RuntimeError):
        HF.cross_entropy(input, target.float())
    with pytest.raises(RuntimeError):
        HF.cross_entropy(input, target, weight=torch.ones(2))
    with pytest.raises(ValueError, match="reduction"):
        HF.cross_entropy(input, target, reduction="bad")
    with pytest.raises(RuntimeError):
        HF.cross_entropy(input, target, label_smoothing=1.1)
    with pytest.raises(RuntimeError):
        HF.cross_entropy(input, torch.ones_like(input), ignore_index=0)
    with pytest.raises(RuntimeError):
        HF.cross_entropy(input, torch.ones_like(input, dtype=torch.int64))


def test_cross_entropy_higher_derivatives() -> None:
    """Keep ordinary first and second derivatives through weighted smoothing."""
    input = torch.tensor([[0.2, -0.3, 1.0]], dtype=torch.float64, requires_grad=True)

    def objective(values: Tensor) -> Tensor:
        """Evaluate the configured compatibility objective."""
        return HF.cross_entropy(
            values,
            torch.tensor([1]),
            weight=values.new_tensor([0.5, 2.0, 1.0]),
            label_smoothing=0.2,
        )

    assert torch.autograd.gradcheck(objective, (input,))
    assert torch.autograd.gradgradcheck(objective, (input,))


@pytest.mark.parametrize("smoothing", [0.0, 0.2])
def test_zero_total_class_weight(smoothing: float) -> None:
    """Match native non-finite mean values and derivatives at zero total weight."""
    input = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1])
    weight = torch.zeros(3, dtype=torch.float64)
    _assert_forward_backward(
        HF.cross_entropy(input, target, weight, label_smoothing=smoothing),
        F.cross_entropy(input, target, weight, label_smoothing=smoothing),
        (input,),
    )


def test_noncontiguous_spatial_classification() -> None:
    """Honor native class-axis layout on strided input tensors."""
    input = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(2, 2, 2, 3)
    input = input.movedim(-1, 1).requires_grad_()
    targets = torch.tensor([[[0, 1], [2, 0]], [[2, 1], [0, 2]]])
    _assert_forward_backward(
        HF.cross_entropy(input, targets, label_smoothing=0.2),
        F.cross_entropy(input, targets, label_smoothing=0.2),
        (input,),
    )


def test_probability_targets_remain_unnormalized() -> None:
    """Match native behavior without silently normalizing user-provided targets."""
    input = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[-0.5, 2.0, 0.1]], dtype=torch.float64, requires_grad=True)
    _assert_forward_backward(
        HF.cross_entropy(input, target, label_smoothing=0.2),
        F.cross_entropy(input, target, label_smoothing=0.2),
        (input, target),
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("probability_target", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_weighted_smoothed_low_precision(
    dtype: torch.dtype, probability_target: bool, autocast: bool
) -> None:
    """Keep native promotion and gradient behavior with combined precision options."""
    input = torch.tensor(
        [[0.1, 0.2, -0.3], [0.3, -0.1, 0.2]], dtype=dtype, requires_grad=True
    )
    target = torch.tensor([0, 2])
    if probability_target:
        target = input.detach().softmax(1).requires_grad_()
    weight = torch.tensor([0.5, 2.0, 1.0], dtype=dtype)
    with torch.autocast("cpu", enabled=autocast):
        actual = HF.cross_entropy(input, target, weight, label_smoothing=0.2)
        expected = F.cross_entropy(input, target, weight, label_smoothing=0.2)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    inputs = (input, target) if probability_target else (input,)
    actual_gradients = torch.autograd.grad(actual, inputs)
    expected_gradients = torch.autograd.grad(expected, inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.02, atol=0.01
        )


def test_probability_mixed_dtype_promotion() -> None:
    """Preserve promotion from probability targets and independently typed weights."""
    input = torch.tensor([[0.1, 0.2, -0.3]], requires_grad=True)
    target = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64, requires_grad=True)
    actual = HF.cross_entropy(input, target, weight, label_smoothing=0.2)
    expected = F.cross_entropy(input, target, weight, label_smoothing=0.2)
    assert actual.dtype == expected.dtype == torch.float64
    _assert_forward_backward(actual, expected, (input, target, weight))


def test_byte_index_targets() -> None:
    """Retain native support for legacy byte class labels."""
    input = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1], dtype=torch.uint8)
    _assert_forward_backward(
        HF.cross_entropy(input, target, label_smoothing=0.2),
        F.cross_entropy(input, target, label_smoothing=0.2),
        (input,),
    )


class _FunctionOverride:
    """Minimal tensor-like boundary object that handles functional dispatch."""

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type[object], ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> Tensor:
        """Return a sentinel without running any tensor computation."""
        return torch.tensor(17.0)


@pytest.mark.parametrize("name", ["softmax", "log_softmax", "cross_entropy"])
def test_tensor_function_override(name: str) -> None:
    """Respect native tensor-like override dispatch before accessing attributes."""
    input = cast(Tensor, _FunctionOverride())
    if name == "cross_entropy":
        actual = HF.cross_entropy(input, torch.tensor([0]))
    else:
        actual = getattr(HF, name)(input, dim=1)
    torch.testing.assert_close(actual, torch.tensor(17.0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_confident_low_precision_backward(dtype: torch.dtype) -> None:
    """Retain representable confident loss and gradients with native output dtype."""
    input = torch.tensor([[10.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)
    actual = HF.cross_entropy(input, torch.tensor([0]))
    gradient = torch.autograd.grad(actual, input)[0]
    assert actual.dtype == dtype
    assert actual > 0
    assert gradient[0, 0] < 0
    assert bool((gradient[0, 1:] > 0).all())


@pytest.mark.parametrize("probability_target", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_autocast_compatibility(
    probability_target: bool, dtype: torch.dtype
) -> None:
    """Check weighted and smoothed CUDA autocast forward/backward parity."""
    input = torch.tensor(
        [[0.1, 0.2, -0.3]], device="cuda", dtype=dtype, requires_grad=True
    )
    target = (
        input.detach().softmax(1)
        if probability_target
        else torch.tensor([1], device="cuda")
    )
    weight = input.new_tensor([0.5, 2.0, 1.0])
    with torch.autocast("cuda", dtype=torch.float16):
        actual = HF.cross_entropy(input, target, weight, label_smoothing=0.2)
        expected = F.cross_entropy(input, target, weight, label_smoothing=0.2)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)
    torch.testing.assert_close(
        torch.autograd.grad(actual, input)[0],
        torch.autograd.grad(expected, input)[0],
        rtol=0.02,
        atol=0.01,
    )
