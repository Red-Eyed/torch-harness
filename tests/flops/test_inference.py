"""Standalone inference profile accuracy, metadata, and state preservation."""

import pytest
import torch
from torch import nn
from typing_extensions import override

from torch_harness import InferenceProfile, profile_inference


class _ProbeModel(nn.Module):
    """Verify inference semantics inside a small classifier with mixed modes."""

    def __init__(self) -> None:
        """Create known parameters and buffers without relying on external assets."""
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout()
        self.calls = 0
        self.fail = False
        self.error = RuntimeError("intentional forward failure")
        self.linear.bias.requires_grad_(False)
        self.norm.eval()

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Assert profiling uses eval and inference modes and execute one forward."""
        self.calls += 1
        assert not torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.modules())
        output = self.dropout(self.norm(self.linear(inputs)))
        assert not output.requires_grad
        if self.fail:
            raise self.error
        return output


@pytest.fixture
def model() -> _ProbeModel:
    """Provide a deterministic classifier with existing parameter gradients."""
    classifier = _ProbeModel()
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.fill_(0.25)
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
    return classifier


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_metrics_and_input_preservation(batch_size: int, dtype: torch.dtype) -> None:
    """Count parameters and whole-batch FLOPs without implicit normalization."""
    linear = nn.Linear(4, 3).to(dtype=dtype)
    linear.bias.requires_grad_(False)
    inputs = torch.ones(batch_size, 4, dtype=dtype, requires_grad=True)
    before = inputs.detach().clone()
    report = profile_inference(linear, args=(inputs,))
    assert isinstance(report, InferenceProfile)
    assert report.parameters == 15
    assert report.trainable_parameters == 12
    assert report.flops.forward_flops == 24 * batch_size
    assert report.gflops == 24 * batch_size / 1e9
    assert report.flops.input_shapes == ((batch_size, 4),)
    assert report.flops.batch_size == batch_size
    assert report.flops.multiply_add_flops == 2
    assert report.torch_version == str(torch.__version__)
    assert report.model_mode == "eval"
    assert report.grad_mode == "no_grad"
    assert inputs.dtype == dtype
    assert inputs.requires_grad
    assert inputs.grad is None
    assert inputs.device.type == "cpu"
    torch.testing.assert_close(inputs, before)


def test_modes_gradients_and_state_are_restored(model: _ProbeModel) -> None:
    """Run once and preserve mixed training modes, parameter identities, and state."""
    flags = tuple(module.training for module in model.modules())
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    profile = profile_inference(model, args=(torch.ones(1, 4),))
    assert profile.parameters == 21
    assert profile.trainable_parameters == 18
    assert model.calls == 1
    assert tuple(module.training for module in model.modules()) == flags
    for parameter, original, gradient in zip(
        model.parameters(), parameters, gradients, strict=True
    ):
        assert parameter is original
        assert parameter.grad is gradient
        if gradient is not None:
            torch.testing.assert_close(gradient, torch.ones_like(gradient))
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()


def test_failure_restores_modes_and_counting(model: _ProbeModel) -> None:
    """Preserve exception identity and restore contexts after a failed forward."""
    flags = tuple(module.training for module in model.modules())
    model.fail = True
    with pytest.raises(RuntimeError, match="intentional forward failure") as caught:
        profile_inference(model, args=(torch.ones(1, 4),))
    assert caught.value is model.error
    assert model.calls == 1
    assert tuple(module.training for module in model.modules()) == flags
    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()
    model.fail = False
    profile = profile_inference(model, args=(torch.ones(1, 4),))
    assert profile.flops.forward_flops == 24


def test_nested_and_keyword_inputs() -> None:
    """Retain shapes from a nested positional input and a keyword tensor."""

    class _NestedModel(nn.Module):
        """Apply a buffered projection to nested inputs."""

        weight: torch.Tensor

        def __init__(self) -> None:
            """Register weights as a buffer to distinguish parameter accounting."""
            super().__init__()
            self.register_buffer("weight", torch.ones(4, 3))

        @override
        def forward(
            self, nested: list[tuple[torch.Tensor]], *, scale: torch.Tensor
        ) -> torch.Tensor:
            """Apply an explicit linear map and scale without trainable weights."""
            return (nested[0][0] @ self.weight) * scale

    profile = profile_inference(
        _NestedModel(),
        args=([(torch.ones(2, 4),)],),
        kwargs={"scale": torch.ones(2, 3)},
    )
    assert profile.parameters == 0
    assert profile.trainable_parameters == 0
    assert profile.flops.forward_flops == 48
    assert profile.flops.input_shapes == ((2, 4), (2, 3))


def test_shared_parameters_are_counted_once() -> None:
    """Deduplicate tied parameter objects while counting each execution."""

    class _SharedModel(nn.Module):
        """Use the same frozen projection twice."""

        def __init__(self) -> None:
            """Tie two registered layers to the same parameter object."""
            super().__init__()
            self.left = nn.Linear(4, 3, bias=False)
            self.right = self.left
            self.left.weight.requires_grad_(False)

        @override
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Execute both projections for a known doubled matrix FLOP count."""
            return self.left(inputs) + self.right(inputs)

    profile = profile_inference(_SharedModel(), args=(torch.ones(1, 4),))
    assert profile.parameters == 12
    assert profile.trainable_parameters == 0
    assert profile.flops.forward_flops == 48


def test_custom_formulas_do_not_change_the_registry() -> None:
    """Allow local counting extensions without affecting subsequent profiles."""

    def formula(*args: object, **kwargs: object) -> int:
        """Return an unmistakable replacement count for matrix multiplication."""
        return 123

    linear = nn.Linear(4, 3, bias=False)
    custom = profile_inference(
        linear, args=(torch.ones(1, 4),), custom_mapping={torch.ops.aten.mm: formula}
    )
    standard = profile_inference(linear, args=(torch.ones(1, 4),))
    assert custom.flops.forward_flops == 123
    assert standard.flops.forward_flops == 24


def test_compiled_models_are_rejected() -> None:
    """Reject opaque execution before attempting model inference."""
    model = nn.Linear(4, 3)
    model.compile(backend="eager")
    with pytest.raises(RuntimeError, match="uncompiled"):
        profile_inference(model, args=(torch.ones(1, 4),))
    assert model.training


def test_higher_order_execution_is_rejected() -> None:
    """Apply the shared execution guard and restore model mode after rejection."""

    class _ConditionalModel(nn.Module):
        """Expose unsafe higher-order counting without executing branches."""

        @override
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Invoke a conditional operator that profiling must reject."""

            def branch(x: torch.Tensor) -> torch.Tensor:
                """Fail if a counting backend attempts to execute this branch."""
                raise AssertionError("branch must not execute")

            return torch.ops.higher_order.cond(
                torch.tensor(True), branch, branch, (inputs,)
            )

    model = _ConditionalModel()
    with pytest.raises(RuntimeError, match="higher-order"):
        profile_inference(model, args=(torch.ones(1),))
    assert model.training
    assert torch.is_grad_enabled()


def test_caller_inference_mode_is_restored() -> None:
    """Profile gradient-bearing inputs safely even inside an inference context."""
    linear = nn.Linear(4, 3)
    inputs = torch.ones(1, 4, requires_grad=True)
    with torch.inference_mode():
        profile = profile_inference(linear, args=(inputs,))
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
    assert profile.flops.forward_flops == 24
    assert inputs.requires_grad
    assert torch.is_grad_enabled()
