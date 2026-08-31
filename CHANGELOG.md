# Changelog

## [Unreleased]

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
