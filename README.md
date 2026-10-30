# torch-harness

An intentionally uncategorized collection of utilities for PyTorch modeling
experiments. It is a toolbox rather than a single framework: utilities are added
as experimental needs arise, without forcing them into an artificial hierarchy.

Each utility is modular and self-contained. It owns its implementation and tests,
avoids assumptions about the surrounding project, and can be used independently
of the other utilities. Optional framework integrations remain isolated so they
do not add dependencies to the core package.

## Imports

All public utilities are available through the root package:

```python
import torch_harness as th

description = th.describe_tensor(tensor)
loss = th.SuperLoss()
model = th.replace_layers(model, old_layer_cls, replacement_factory)
counter = th.FlopCounter()  # Requires torch-harness[lightning].
th.write_text_graph(onnx_model, output_path)  # Requires torch-harness[onnx].
```

Lightning and ONNX exports load on first access. Importing `torch_harness` and
using core utilities does not require either extra. Accessing an optional utility
without its dependency raises `ModuleNotFoundError` with the extra to install.
The aliases retain the original objects and type annotations, and appear in
`dir(th)` without loading the integrations. Existing namespace imports such as
`from torch_harness.lightning import FlopCounter` remain supported. A wildcard
import requests every export and therefore requires both optional integrations.

## Tensor descriptions

`describe_tensor` returns a string of statistics for tensors nested inside
mappings and iterables, or registered parameters and buffers inside PyTorch
modules, with paths identifying each tensor:

```python
import torch

from torch_harness import describe_tensor

description = describe_tensor(
    {"features": [torch.tensor([1.0, 2.0, float("nan")])]},
    percentiles=(0.25, 0.5, 0.75),
)
print(description, end="")

description = describe_tensor(torch.nn.Sequential(torch.nn.Linear(2, 3)))
# Contains root.0.weight and root.0.bias.
```

Modules, including `Sequential`, `ModuleList`, and `ModuleDict`, are inspected
without running a forward pass or invoking state-dict hooks. Registered buffers
are included even when non-persistent. Shared parameters and shared buffers are
each reported once per module traversal, under their first registered name.
Unregistered tensor attributes and absent parameters or buffers are omitted.
Ordered mappings can contain tensors, modules, or nested mixtures of both.

Each description includes shape, original dtype and device, element count,
non-finite count (`null_count`), and finite-value mean, population standard
deviation, minimum, quantiles, and maximum. Empty or entirely non-finite tensors
include metadata only. Float64 precision is retained; other real dtypes use
float32 calculations. Large integers can therefore round. Inputs and gradients
remain unchanged.

Percentiles are fractions in `[0, 1]`; an empty tuple omits quantiles. Strings,
bytes, and non-tensor leaves are ignored. Iterators are consumed, and containers
must be finite and acyclic. Complex, sparse, quantized, and meta tensors are
rejected. Statistics use all finite values without sampling and can allocate
substantial temporary storage or synchronize accelerators. Quantile support and
size limits follow PyTorch. The function writes no console output. Use
`print(describe_tensor(obj), end="")` to display it, or pass the returned string
to your logger. The result is an empty string when no tensors are found.

## Adaptive loss balancing

`LossBalancer` adapts named auxiliary weights from shared gradient norms while
keeping the primary loss weight exactly one:

```python
from torch_harness.losses import LossBalancer, LossTarget, weighted_loss

balancer = LossBalancer(
    reference="primary",
    adaptive={"auxiliary": LossTarget(ratio=0.3, initial_weight=0.1)},
)
weights = balancer.update(losses, shared=shared_features, step=global_step)
total_loss = weighted_loss(losses, weights)
```

It supports independent targets, fixed weights, optional losses, periodic EMA
updates, checkpoint continuation, and synchronized DDP activation measurements.
See the [component documentation](src/torch_harness/losses/adaptive_balancing/README.md)
for stability controls, diagnostics, and training integration. Run
`uv run examples/adaptive_loss_balancing.py` for a synthetic optimization example.

## Uncertainty weighting

`UncertaintyWeighting` learns positive weights for a vector of peer task losses:

```python
import torch

from torch_harness.losses import UncertaintyWeighting

weighting = UncertaintyWeighting(num_tasks=3)
task_losses = torch.stack((classification_loss, depth_loss, normal_loss))
loss = weighting(task_losses)
weights_for_logging = weighting.effective_weights
```

Each effective weight starts near one and adapts jointly with the model. The
module preserves gradients to the input task losses and its uncertainty
parameters, while the weights exposed for logging are detached. See the
[component documentation](src/torch_harness/losses/uncertainty_weighting/README.md)
for the exact objective, optimizer wiring, numerical behavior, and limitations.

## SuperLoss

`SuperLoss` applies robust curriculum weighting to any unreduced task loss:

```python
from math import log

import torch.nn.functional as F

from torch_harness.losses import SuperLoss

criterion = SuperLoss(threshold=log(10), regularization=1.0)
task_loss = F.cross_entropy(logits, targets, reduction="none")
loss = criterion(task_loss)
```

It computes the closed-form optimal sample confidence from the SuperLoss paper,
upweighting easy samples and downweighting hard samples. The implementation is
task-agnostic, runs entirely in PyTorch, and has no SciPy or per-sample-state
dependency. See the [component documentation](src/torch_harness/losses/super_loss/README.md)
for parameter guidance and numerical details.

## Stable softmax and cross-entropy

`stable_cross_entropy` preserves the small loss and target gradient of confident
predictions. `stable_distillation_cross_entropy` accepts student and teacher
logits directly and preserves small student gradients near teacher agreement:

```python
import torch_harness as th

loss = th.stable_cross_entropy(logits, targets, dim=-1)
distillation = th.stable_distillation_cross_entropy(student_logits, teacher_logits)
```

`stable_log_softmax` and `stable_softmax` expose the underlying normalization.
These opt-in functions implement [Barron et al. (2026)](https://arxiv.org/abs/2610.31642)
using ordinary PyTorch operations. The default class axis is last; low-precision
inputs compute and return FP32. Distillation detaches the teacher and supports
first-order student gradients only. Compiled distillation requires backend-specific
numerical validation. See the [component documentation](src/torch_harness/losses/stable_cross_entropy/README.md)
for contracts, limitations, and the forward/backward benchmark.

## SafeBatchNorm

`SafeBatchNorm` rejects a non-finite activation before it can corrupt BatchNorm
running statistics.

```python
from torch_harness.layers import SafeBatchNorm2d

normalization = SafeBatchNorm2d(64)
```

`SafeBatchNorm1d`, `SafeBatchNorm2d`, and `SafeBatchNorm3d` directly inherit
their matching PyTorch classes. They retain the native constructor, state-dict
layout, and type identity. A non-finite input raises `FloatingPointError`, which
a fault-tolerant training loop can catch to skip the step.

`nn.SyncBatchNorm` is intentionally unsupported because a rank-local failure
before its collective could deadlock the other ranks.

## Runtime layer replacement

`replace_layers` recursively transforms existing models through an explicit
replacement factory:

```python
from torch import nn

from torch_harness.runtime import replace_layers

model = replace_layers(
    model,
    old_layer_cls=nn.SiLU,
    replacement_factory=lambda silu: nn.ReLU(inplace=silu.inplace),
)
```

The traversal includes nested containers such as `Sequential`. Run replacement
before constructing the optimizer, distributed wrappers, or a compiled model.

## Fault-tolerant training steps

`FaultTolerantTrainingStep` skips occasional exceptions and NaN or infinite losses
before Lightning automatic optimization runs backward or updates the optimizer:

```python
from datetime import timedelta

from lightning.pytorch import Trainer
from torch_harness.lightning import FaultTolerantTrainingStep

fault_tolerance = FaultTolerantTrainingStep(
    max_faults=3,
    fault_window=timedelta(hours=1),
)
trainer = Trainer(callbacks=[fault_tolerance])
```

This example skips the first three faults in any rolling one-hour window. A fourth
fault within that window is logged through Loguru and propagated. Exceptions retain
their original type and traceback; repeated non-finite losses raise
`NonFiniteLossError`.

Install the Lightning integration with `torch-harness[lightning]`.

The callback intentionally supports only single-process automatic optimization.
Errors raised during backward or `optimizer.step` happen after the recoverable
boundary and are propagated because an optimizer update may already be partial.

## Model structures in files

`FileModelStructure` saves the recursive structure produced by `print(model)` as
readable UTF-8 text when fitting starts:

```python
from pathlib import Path

from lightning.pytorch import Trainer
from torch_harness.lightning import FileModelStructure

structure = FileModelStructure(
    output_path=Path("artifacts/model-structure.txt"),
)
trainer = Trainer(callbacks=[structure])
```

The callback includes the complete registered module hierarchy without requiring
example inputs or running a forward pass. It creates parent directories, replaces
an existing structure file, and writes only from the global-zero process. Runtime
tensor operations that are not registered modules do not appear.

Run `uv run examples/file_model_structure.py` to generate an inspectable example
at `examples/model-structure.txt`.

## Inference profiling

Use `profile_inference` without Lightning to produce parameter and GFLOP metrics
for a fixed reference input:

```python
import torch
import torch_harness as th

model = torch.nn.Linear(4, 3)
profile = th.profile_inference(model, args=(torch.ones(1, 4),))
parameters = profile.parameters
gflops = profile.gflops
```

The function executes exactly one eval-mode forward with gradients disabled,
restores every module's prior training flag, and returns an immutable
`InferenceProfile`. It includes total and trainable parameter counts, input
shapes, batch size, the PyTorch version, execution mode, and the convention
**one multiply-add = two FLOPs**. Shared parameters count once; buffers do not
count as parameters. GFLOPs use decimal billions and cover the entire input
batch, without rounding or automatic per-sample normalization.

Use the same reference workload across models and batch size one for per-sample
paper figures. Unsupported ordinary operators may count zero; compiled and
higher-order execution is rejected. See the
[profiling documentation](src/torch_harness/flops/README.md) for keyword inputs,
custom formulas, state-restoration guarantees, and limitations.

## Forward FLOPs

`FlopCounter` measures only the first model forward in the first real training
batch, once per fit, without replaying the model:

```python
from lightning.pytorch import Trainer
from torch_harness.lightning import FlopCounter

counter = FlopCounter()
trainer = Trainer(callbacks=[counter])
```

Install `torch-harness[lightning]`. By default, `training_step` must invoke the
LightningModule as `self(inputs)`. If it calls an inner model directly, select
it with `FlopCounter(module=model.network)`. Loss outside forward, backward,
optimizer updates, and later forwards are excluded. Reports include total forward
FLOPs, observed input shapes, the assumed batch size, and the convention that one
multiply-add counts as two FLOPs. Counts describe the whole observed input; use
batch size one for a per-image paper figure. The model's mode is unchanged.

Partitioned window attention counts from its actual operator shapes; custom
formulas can extend coverage. Unsupported operations may contribute zero.
Compiled modules and higher-order operators are rejected during measurement.

See the [counting documentation](src/torch_harness/flops/README.md) for the forward
boundary, paper reporting, distributed scope, extension contracts, and limitations.

## Human-readable ONNX graphs

Install `torch-harness[onnx]` to save module summaries suitable for Git diffs
and Meld:

```python
from pathlib import Path

import onnx

from torch_harness.onnx import write_text_graph

model = onnx.load("model.onnx")
write_text_graph(model, Path("artifacts/model.onnx.txt"))
```

Summaries show operation signatures and counts, attributes, small constants,
parameter shapes, and input/output dimensions. They omit temporary node indexes
and weight payloads, making RMSNorm decomposition versus `RMSNormalization`
visible without renumbering unrelated operations. Internal wiring is omitted.

See the [ONNX documentation](src/torch_harness/onnx/README.md) for source module
capture, shape inference, and format limitations. Run
`uv run --extra onnx examples/onnx_text_graph.py` for a small example.

## Planned utilities

- A Lightning `PreciseBN` callback for recomputing BatchNorm running statistics.
- A Lightning mixin that manages schedule-free AdamW train/evaluation state.

Each integration will be isolated so users only install the frameworks they need.

## License

`torch-harness` is released under the MIT License.
