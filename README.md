# torch-harness

An intentionally uncategorized collection of utilities for PyTorch modeling
experiments. It is a toolbox rather than a single framework: utilities are added
as experimental needs arise, without forcing them into an artificial hierarchy.

Each utility is modular and self-contained. It owns its implementation and tests,
avoids assumptions about the surrounding project, and can be used independently
of the other utilities. Optional framework integrations remain isolated so they
do not add dependencies to the core package.

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

## Planned utilities

- A Lightning `PreciseBN` callback for recomputing BatchNorm running statistics.
- A Lightning mixin that manages schedule-free AdamW train/evaluation state.

Each integration will be isolated so users only install the frameworks they need.

## License

`torch-harness` is released under the MIT License.
