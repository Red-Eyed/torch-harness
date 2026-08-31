# torch-utils

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
from torch_utils.layers import SafeBatchNorm2d

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

from torch_utils.runtime import replace_layers

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
from torch_utils.lightning import FaultTolerantTrainingStep

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

Install the Lightning integration with `torch-utils[lightning]`.

The callback intentionally supports only single-process automatic optimization.
Errors raised during backward or `optimizer.step` happen after the recoverable
boundary and are propagated because an optimizer update may already be partial.

## Planned utilities

- A Lightning `PreciseBN` callback for recomputing BatchNorm running statistics.
- A Lightning mixin that manages schedule-free AdamW train/evaluation state.

Each integration will be isolated so users only install the frameworks they need.

## License

`torch-utils` is released under the MIT License.
