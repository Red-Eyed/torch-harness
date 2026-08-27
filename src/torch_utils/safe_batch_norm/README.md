# SafeBatchNorm

The safe BatchNorm family directly inherits PyTorch's standard BatchNorm
classes and rejects a faulty input before it can change the running mean,
running variance, or batch counter.

```python
from torch_utils import SafeBatchNorm2d

normalization = SafeBatchNorm2d(64)
```

A forward input containing a NaN or infinity raises `FloatingPointError`. A
fault-tolerant training loop can catch that exception and skip the step without
retaining corrupted BatchNorm statistics.

## Architecture

`SafeBatchNorm1d`, `SafeBatchNorm2d`, and `SafeBatchNorm3d` inherit their
matching PyTorch classes, preserving native constructor signatures, state-dict
keys, and type identity. A shared helper owns only the finite-input guard;
PyTorch continues to own normalization, gradients, and dimension validation.

The guard converts a scalar tensor to `bool`, which introduces a device-to-host
synchronization when the input is on an accelerator.

`SyncBatchNorm` is intentionally unsupported. A local guard could make one rank
raise while other ranks enter the synchronization collective and wait forever.
