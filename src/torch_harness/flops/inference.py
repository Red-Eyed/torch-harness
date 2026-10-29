"""Standalone inference profiling for reproducible parameter and FLOP tables."""

from collections.abc import Callable, Mapping

import torch

from torch_harness.flops._measurement import (
    count_forward,
    forward_report,
    input_shapes,
    validate_eager,
)
from torch_harness.flops.report import InferenceProfile


def profile_inference(
    model: torch.nn.Module,
    *,
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object] | None = None,
    custom_mapping: Mapping[object, Callable[..., int]] | None = None,
) -> InferenceProfile:
    """Measure parameters and theoretical FLOPs for one eval-mode inference call.

    Runs model(*args, **kwargs) exactly once under torch.no_grad(), with
    all registered modules temporarily in eval mode. Each module's original
    training flag is restored on success or failure, including mixed modes.
    Parameters, gradients, inputs, devices, and dtypes are not changed by the
    profiler. Model-defined forward and train/eval side effects still occur;
    arbitrary mutations by user code cannot be rolled back. Outputs are not
    retained, and no extra warmup or backward pass is performed. Inference mode
    is temporarily disabled because PyTorch's module tracker requires ordinary
    tensors when a model input already has requires_grad=True. The caller's
    gradient and inference modes are restored afterward.

    Counts cover the whole supplied input batch, using two FLOPs per multiply-add
    and decimal GFLOPs (10**9). Unsupported ordinary operators contribute zero;
    results do not certify complete coverage. Compiled and higher-order execution
    is rejected. Input metadata supports nested tuples, lists, and mappings.
    Batch size assumes the first non-scalar input's leading dimension. Use batch
    size one for per-sample paper results unless normalization is justified.

    Args:
        model: Eager model whose parameters and inference execution to profile.
        args: Positional arguments for the reference forward call.
        kwargs: Keyword arguments for the reference forward call.
        custom_mapping: Local operator FLOP formulas following PyTorch's
            FlopCounterMode contract; the global registry is not modified.

    Returns:
        Immutable parameter counts, forward accounting metadata, decimal GFLOPs,
        and the PyTorch version. Shared parameter objects are counted once;
        buffers are excluded and trainable counts use requires_grad.

    Raises:
        RuntimeError: If compiled or unsupported higher-order execution is used,
            or PyTorch does not provide FlopCounterMode. Model errors propagate.
    """
    validate_eager(model)
    keyword_inputs = dict(kwargs or {})
    shapes = tuple(input_shapes((args, keyword_inputs)))
    parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameters += size
        if parameter.requires_grad:
            trainable_parameters += size
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        with (
            torch.inference_mode(False),
            torch.no_grad(),
            count_forward(custom_mapping or {}) as counter,
        ):
            model(*args, **keyword_inputs)
        report = forward_report(counter, shapes)
    finally:
        for module, training in modes:
            module.training = training
    return InferenceProfile(
        parameters=parameters,
        trainable_parameters=trainable_parameters,
        flops=report,
        torch_version=str(torch.__version__),
    )
