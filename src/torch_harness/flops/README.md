# One-shot forward FLOPs

`FlopCounter` measures the first model forward in the first real Lightning
training batch, once per fit. It observes existing execution without replaying
inputs, changing model mode, or performing an additional forward pass:

```python
import torch_harness as th
from lightning.pytorch import Trainer

counter = th.FlopCounter()
trainer = Trainer(callbacks=[counter])
# trainer.fit(model, train_dataloaders=loader)
if isinstance(counter.latest_report, th.FlopReport):
    report = counter.latest_report
    print(report.forward_flops, report.input_shapes, report.batch_size)
```

Install `torch-harness[lightning]`. The callback additionally requires a PyTorch
version providing `torch.utils.flop_counter.FlopCounterMode`; other utilities
remain importable without that module.

## Measurement boundary

By default, hooks observe `LightningModule.__call__`, so training_step should
invoke the model as `self(inputs)`. For code that calls a nested model directly,
select it explicitly:

```python
counter = th.FlopCounter(module=model.network)
```

Calling `.forward(...)` directly bypasses PyTorch hooks; call the module instead.
Counting starts at the selected module's first forward pre-hook and stops in its
forward post-hook. The callback then removes its hooks and dispatch modes. A
second invocation, even in the same training step, is not measured. Loss
computation outside forward, backward, checkpoint recomputation during backward,
optimizer updates, validation, and sanity checks are excluded. Work deliberately
placed inside forward is included. Existing pre-hooks registered earlier run
before measurement; existing local post-hooks run afterward. Global hooks and
nested module hooks may contribute work within the observed boundary.

The selected forward must complete in the first training batch. If it is never
called or a forward failure is swallowed, the callback warns and leaves the
report unmeasured; it does not substitute whole-training-step counts or retry on
later batches. Propagated failures retain their original exception and clean up
counting state. A successful report survives a later failure. Each subsequent
fit resets measurement; callback results are not checkpointed.

## Paper reporting

`latest_report` starts as `UnmeasuredFlops(reason="no_complete_forward")`.
A successful `FlopReport` contains:

- `forward_flops`: theoretical arithmetic for the complete observed input.
- `input_shapes`: tensor shapes in positional then keyword encounter order,
  recursively through tuples, lists, and mappings. No input tensors are retained.
- `batch_size`: the leading dimension of the first non-scalar tensor input,
  or one if none exists. This is a batch-first convention, not semantic inference;
  verify it for sequence-first inputs or other layouts.
- `operator_flops`: sorted Global operator totals. Module rows are not summed
  because parent and child attribution overlap.
- `multiply_add_flops`: two, meaning one multiplication plus one addition counts
  as two FLOPs. A paper counting multiply-accumulates as one uses a different
  convention; the complete total cannot generally be divided by two because
  not every operator represents multiply-accumulates.

For a per-image paper figure, use an input batch of size one or justify the
normalization for your model. The callback never silently divides by batch size.
It observes the current training mode; it does not claim to measure eval-mode
inference. Report input resolution/sequence lengths, batch size, model mode,
PyTorch version, counting convention, and known operator-coverage limitations.
No claim of exact hardware instruction count or full operator coverage is made.

Reports are rank-local, without collectives or world-size multiplication.
On rank zero the callback writes the report through Loguru and, when
`log_metrics=True`, emits one `flops/forward` record directly to each configured
Lightning logger. This avoids Lightning's per-step metric cache repeating the
value. It does not populate `trainer.callback_metrics`. Logger values are floats;
use the report for exact integer counts. Other ranks retain their local reports.
Multi-device and accelerator strategies have not been validated by the CPU tests.

## Windowed attention and custom operators

Partitioned windows expose smaller attention shapes. For batch size B, H heads,
N tokens, W tokens per window, and equal head dimension D, QK and AV matrix
products cost `4 * B * H * N * W * D` FLOPs when windows divide N evenly.
Projections and other operators are separate. A mask applied after dense global
attention does not reduce its matrix cost; padding counts as executed work.

Pass `custom_mapping={operator_packet: formula}` to extend or override formulas
without modifying PyTorch's global registry. Keys such as `torch.ops.aten.mm`
identify operator packets. Tensor arguments become shapes, non-tensor arguments
pass through, and `out_shape` is provided by keyword. A formula accepts the
relevant arguments and returns an integer FLOP count. Ordinary custom operators
still execute normally; no backward formulas are needed for this callback.

Unsupported ordinary operators contribute zero. Dense formulas may overestimate
masked or sparse arithmetic. Memory traffic, device throughput, wall-clock time,
and dtype-specific hardware execution are not represented.

Compiled modules are rejected at fit start. Higher-order operators, including
Triton wrappers and control flow, are rejected during measurement: some PyTorch
FLOP handlers skip kernels or execute extra branches. A formula does not bypass
this guard. Fused or compiled backends need separate execution validation.

## Functional core and validation

`torch_harness.flops.summarize_flops` is independent of Lightning and PyTorch. It
validates counts and shapes and constructs the immutable forward report. The
callback owns model hooks, dispatch contexts, one-shot lifecycle, and logging.

CPU tests verify analytical attention arithmetic, exact forward boundaries,
multiple calls/batches/epochs, gradient accumulation, manual optimization,
checkpoint recomputation exclusion, logger emission, custom formulas, exception
cleanup, and preservation of model updates against an uninstrumented baseline.
They validate mathematical accounting and lifecycle behavior, not hardware work.
