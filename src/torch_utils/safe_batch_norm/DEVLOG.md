# SafeBatchNorm development log

## 0.2.0

### Motivating observation

A fault-tolerant training loop can reject non-finite model outputs, but that
check occurs after BatchNorm has already updated its persistent running state
from the faulty activation.

### Decision

Directly inherit BatchNorm1d, BatchNorm2d, and BatchNorm3d, and raise
`FloatingPointError` when a shared pre-forward input check finds a NaN or
infinity. Checking before BatchNorm prevents any running-state mutation and
leaves PyTorch's forward and backward implementations untouched for finite
inputs. Inheritance preserves native constructors, state-dict keys, and type
identity.

Use an explicit exception rather than `assert`, because optimized Python can
remove assertions. Synchronized BatchNorm is excluded because a rank-local
early failure could deadlock peers already entering its collective.

### Verified result

Parameterized tests against BatchNorm1d, BatchNorm2d, and BatchNorm3d verify
that finite outputs, running-state updates, input gradients, and affine
parameter gradients match native BatchNorm. NaN and infinity inputs raise
before the running mean, running variance, or batch counter changes. Evaluation
mode and disabled running statistics are covered as well.

### Unproven claims

The per-layer scalar check introduces a device-to-host synchronization on
accelerators; its training-throughput cost has not been benchmarked.
