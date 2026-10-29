# torch-harness

An intentionally uncategorized collection of utilities for PyTorch modeling
experiments. It is a toolbox rather than a single framework: utilities are added
as experimental needs arise, without forcing them into an artificial hierarchy.

Each utility is modular and self-contained. It owns its implementation and tests,
avoids assumptions about the surrounding project, and can be used independently
of the other utilities. Optional framework integrations remain isolated so they
do not add dependencies to the core package.

## Tensor descriptions

`describe_tensor` prints statistics for tensors nested inside mappings and
iterables, with paths identifying each tensor:

```python
import torch

from torch_harness import describe_tensor

describe_tensor(
    {"features": [torch.tensor([1.0, 2.0, float("nan")])]},
    percentiles=(0.25, 0.5, 0.75),
)
```

Each description includes shape, original dtype and device, element count,
non-finite count (`null_count`), and finite-value mean, population standard
deviation, minimum, quantiles, and maximum. Empty or entirely non-finite tensors
print metadata only. Float64 precision is retained; other real dtypes use
float32 calculations. Large integers can therefore round. Inputs and gradients
remain unchanged.

Percentiles are fractions in `[0, 1]`; an empty tuple omits quantiles. Strings,
bytes, and non-tensor leaves are ignored. Iterators are consumed, and containers
must be finite and acyclic. Complex, sparse, quantized, and meta tensors are
rejected. Statistics use all finite values without sampling and can allocate
substantial temporary storage or synchronize accelerators. Quantile support and
size limits follow PyTorch. Descriptions are written to standard output.

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

## Training FLOPs

`FlopCounter` measures selected real training batches through PyTorch's operator
formulas, without replaying the model:

```python
from lightning.pytorch import Trainer
from torch_harness.lightning import FlopCounter

counter = FlopCounter(every_n_batches=100, max_batches=5)
trainer = Trainer(callbacks=[counter])
```

Install `torch-harness[lightning]`. Reports separate batch, backward, and
non-backward FLOPs per rank. Partitioned window attention is counted from its
actual operator shapes; custom operator formulas can extend coverage. Unsupported
operations may contribute zero. Compiled modules and higher-order operators are
unsupported and rejected during measurement.

See the [counting documentation](src/torch_harness/flops/README.md) for sampling,
manual optimization, distributed scope, extension contracts, and limitations.

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
