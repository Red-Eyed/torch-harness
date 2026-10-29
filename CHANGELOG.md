# Changelog

## [0.15.0]

### Backwards Incompatible Changes

- Make `FlopCounter` measure only the first model forward in the first training
  batch, once per fit. Remove `every_n_batches` and `max_batches`; use
  `FlopCounter()` instead. Loss outside forward, backward, and optimizer work
  are excluded. Select `module=model.network` if training calls an inner model
  instead of the LightningModule through `self(...)`.
- Replace `FlopReport.batch_flops`, `backward_flops`, and `non_backward_flops`
  with `forward_flops`, `input_shapes`, `batch_size`, and `multiply_add_flops`.
  Remove `summarize_flops(backward_flops=...)`; input metadata is supplied via
  `input_shapes=...`. The unmeasured reason is now `no_complete_forward`.
- Replace batch and backward logger metrics with a single `flops/forward`
  record sent directly to configured loggers. It does not populate Lightning's
  callback metrics or repeat across training steps.

## [0.14.0]

### Backwards Incompatible Changes

- Change `describe_tensor` to return its formatted description as a string
  instead of printing and returning `None`. To preserve console output, replace
  `describe_tensor(obj)` with `print(describe_tensor(obj), end="")`. Formatting
  and traversal are unchanged; inputs without tensors return an empty string.

## [0.13.1]

### Bug Fixes

- Fix `describe_tensor` silently ignoring modules in `Sequential`, `ModuleList`,
  `ModuleDict`, and mappings. Inspect registered parameters and buffers using
  named paths without running a forward pass or modifying model state.

## [0.13.0]

### New Features

- Expose all public utilities through `import torch_harness as th`, including
  runtime transformations, FLOP reports, Lightning callbacks, and ONNX helpers.
  Lightning and ONNX exports load on demand and identify the required extra
  when its dependency is missing. Existing namespace imports remain supported.

## [0.12.0]

### New Features

- Add `describe_tensor` to print finite-value statistics for tensors in nested
  mappings and iterables, including percentile rows and non-finite counts.
  Preserve float64 precision and input gradients; reject invalid percentiles
  and unsupported complex, sparse, quantized, or meta tensors explicitly.

## [0.11.0]

### New Features

- Add `FlopCounter`, a Lightning callback sampling real training batches with
  per-rank batch, backward, and non-backward FLOP reports and local custom
  operator formulas. Counting uses PyTorch's theoretical operator formulas;
  unsupported ordinary operations may contribute zero. Compiled modules and
  higher-order operators are rejected to preserve execution semantics.
- Add framework-independent immutable FLOP reports and pure report construction.

## [0.10.1]

### Developers

- Add `just build` to build wheel and source distributions, and `just all` to
  install locked dependencies, run the CI quality checks, and build artifacts.

## [0.10.0]

### New Features

- Add the optional `torch-harness[onnx]` integration with `format_text_graph`,
  `collect_source_scopes`, and `write_text_graph`. Summaries group operations by
  source module and show signatures, counts, attributes, constants, and tensor
  dimensions without temporary node indexes or weight payloads.
- Infer shapes when writing text summaries, preserving the caller's ONNX model
  and keeping ONNX outside the base package dependencies.

## [0.9.0]

### Backwards Incompatible Changes

#### Package identity

##### Rename the distribution and import namespace

The distribution is now `torch-harness`, and Python imports move from
`torch_utils` to `torch_harness`. Replace the old package name in dependency
configuration and update imports when upgrading.

### New Features

- Add `FileModelStructure` to save the recursive structure produced by
  `print(model)` as a readable UTF-8 file when fitting starts.
- Add `SuperLoss` for applying robust curriculum weighting to arbitrary unreduced
  PyTorch losses, with standard `none`, `mean`, and `sum` reductions.
- Add `UncertaintyWeighting` to learn positive homoscedastic uncertainty weights
  for a rank-one tensor of peer task losses, with detached monitoring weights and
  float32 calculation under autocast.

## [0.4.1] - 2026-08-31

### Documentation

- Clarify that `torch-utils` is a modular toolbox for independent PyTorch modeling
  utilities rather than a single framework or fixed component hierarchy.

## [0.4.0] - 2026-08-28

### New Features

- Add `FaultTolerantTrainingStep` for single-process Lightning automatic
  optimization. It skips exceptions and non-finite losses within a configurable
  rolling `timedelta` window, logs tolerated faults through Loguru, and propagates
  the original exception once the fault budget is exhausted.
- Add the optional `torch-utils[lightning]` installation extra. Exceptions raised
  during backward or `optimizer.step` remain outside the recoverable boundary and
  are propagated.

## [0.3.0] - 2026-08-28

### New Features

- Add `replace_layers` for recursively replacing exact layer types while preserving
  shared-module aliases.

## [0.2.0] - 2026-08-27

### New Features

- Add `SafeBatchNorm1d`, `SafeBatchNorm2d`, and `SafeBatchNorm3d` to reject
  non-finite inputs before they can corrupt running statistics.

## [0.1.0] - 2026-08-27

### Developers

- Establish the typed `torch-utils` package, linting, tests, continuous integration,
  and portable wheel builds.
