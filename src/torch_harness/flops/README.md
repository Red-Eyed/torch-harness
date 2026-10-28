# Training FLOP measurement

`FlopCounter` samples actual Lightning training batches without replaying forward
passes or constructing example inputs:

```python
from lightning.pytorch import Trainer
from torch_harness.flops import FlopReport
from torch_harness.lightning import FlopCounter

counter = FlopCounter(every_n_batches=100, max_batches=5)
trainer = Trainer(callbacks=[counter])
# trainer.fit(model, train_dataloaders=loader)
if isinstance(counter.latest_report, FlopReport):
    report = counter.latest_report
    print(report.batch_flops, report.backward_flops)
```

Install `torch-harness[lightning]`. This callback additionally requires a PyTorch
version providing `torch.utils.flop_counter.FlopCounterMode`; other utilities
remain importable without that module. Measurement is for eager execution.

## What is counted

The callback activates PyTorch's operator counter between its training batch-start
and batch-end hooks, on each rank independently. `latest_report` contains exact
integer totals under the backend's registered formulas:

- `batch_flops`: all supported arithmetic observed during the sampled batch.
- `backward_flops`: arithmetic between Lightning's before/after-backward hooks,
  summed over backward calls, including activation-checkpoint recomputation.
- `non_backward_flops`: the remainder, including forward, loss computation, and
  supported optimizer operations. This is deliberately not named forward FLOPs.
- `operator_flops`: sorted global operator totals. Module totals are not summed,
  since parent and child attribution overlap.

`latest_report` starts as `UnmeasuredFlops(reason="no_complete_batch")`. The latest
complete report is retained when a later batch fails. Reports and sampling limits
reset for each fit and are not checkpointed. No per-batch history is retained.
The interval uses the epoch-local batch index; `max_batches` spans the entire fit.
Unsampled batches have no active counting mode.

By default, the callback sends `flops/batch`, `flops/backward`, and
`flops/non_backward` to the Lightning logger and writes the report through Loguru
on rank zero. Set `log_metrics=False` to disable Lightning metric logging. Logged
metrics are floating point; use the report for exact large integer counts.

Automatic optimization, gradient accumulation, and manual optimization via
`self.manual_backward` are supported. Calling raw `loss.backward()` in manual
optimization contributes to the batch total but bypasses Lightning's backward
hooks, so it cannot be classified as backward work. Data loading, validation,
communication, and work in hooks outside the measurement interval are excluded.
Callback ordering can affect included hook arithmetic. Optimizer closure retries
are counted as executed. Counts describe a microbatch, not an accumulated update.

Distributed counts are per rank, without collectives or multiplication by world
size. Logger metrics describe rank zero; this does not imply equal work across
ranks or a global total. Multi-device and accelerator strategies have not been
validated by this component's CPU tests.

## Windowed attention and custom operators

Partitioned windows naturally expose smaller attention shapes. For batch size B,
H heads, N tokens, W tokens per window, and head dimension D, the QK and AV matrix
products cost `4 * B * H * N * W * D` FLOPs when windows divide N evenly and Q, K,
and V have equal head dimensions. Projection and softmax work are separate.
Applying a mask after dense global attention does not reduce its matrix cost.
Padding counts as executed work.

Pass `custom_mapping={operator_packet: formula}` to override or extend PyTorch's
formulas without modifying the global registry. Keys such as `torch.ops.aten.mm`
identify operator packets. Formula call signatures follow PyTorch: tensor inputs
are replaced by shapes, non-tensor arguments pass through, and `out_shape` is
provided as a keyword. The formula must accept the relevant operator arguments
and return an integer FLOP count. Supply separate formulas for custom backward
operators when applicable. Ordinary custom operators still execute normally.

Unsupported ordinary operators contribute zero; reports do not certify complete
coverage. Dense formulas may overestimate masked attention or sparse arithmetic.
Dtype, memory movement, kernel fusion, device throughput, and wall-clock time
are not represented. One multiply plus one addition counts as two FLOPs.

Compiled modules are rejected at fit start. Higher-order operators, including
Triton wrappers and control flow, are rejected during measurement: some PyTorch
FLOP handlers skip kernel execution or evaluate extra branches, which would alter
real training. A formula does not bypass this guard. Opaque ordinary operators
must likewise preserve execution under PyTorch's dispatcher. Fused and compiled
backends require separate validation before support can be claimed.

## Functional core

`torch_harness.flops.summarize_flops` is independent of Lightning and PyTorch. It
validates a mapping of global operator names to counts and returns an immutable
`FlopReport`. The Lightning callback owns dispatch contexts, sampling, lifecycle
cleanup, and logging. It discards incomplete measurements on propagated failures
and removes counting hooks on exception or teardown.

## Validation

CPU tests compare partitioned attention against analytical matrix counts and
exercise real Lightning automatic/manual optimization, accumulation, sampling,
local formula overrides, and exception cleanup. They validate mathematical
counting and lifecycle behavior, not hardware instruction counts.
