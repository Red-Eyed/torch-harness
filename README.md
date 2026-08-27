# torch-utils

Reliable, focused utilities for PyTorch and Lightning projects.

## SafeBatchNorm

`SafeBatchNorm` rejects a non-finite activation before it can corrupt BatchNorm
running statistics.

```python
from torch_utils import SafeBatchNorm2d

normalization = SafeBatchNorm2d(64)
```

`SafeBatchNorm1d`, `SafeBatchNorm2d`, and `SafeBatchNorm3d` directly inherit
their matching PyTorch classes. They retain the native constructor, state-dict
layout, and type identity. A non-finite input raises `FloatingPointError`, which
a fault-tolerant training loop can catch to skip the step.

`nn.SyncBatchNorm` is intentionally unsupported because a rank-local failure
before its collective could deadlock the other ranks.

## Planned utilities

- A Lightning `PreciseBN` callback for recomputing BatchNorm running statistics.
- A Lightning mixin that manages schedule-free AdamW train/evaluation state.

Each integration will be isolated so users only install the frameworks they need.

## License

`torch-utils` is released under the MIT License.
