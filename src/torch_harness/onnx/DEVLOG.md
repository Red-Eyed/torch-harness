# ONNX summary development log

## 0.10.0

### Motivating observation

Raw ONNX text and generated tensor counters make model diffs noisy. Replacing
decomposed normalization with RMSNormalization should change the relevant module
summary without renumbering unrelated operations.

### Decision

Port the deterministic ONNX module summary as an optional integration. Keep a
pure formatter separate from shape inference and filesystem I/O. Sort modules
and operation signatures, count identical signatures, retain attributes and small
constants, and show parameter and operation dimensions. Omit generated node IDs,
weight payloads, internal tensor wiring, and operation order.

Preserve source scopes through a tensor-name mapping captured before optimization
removes exporter metadata. Use parameter paths and a limited normalization-chain
heuristic when source scopes are unavailable. ONNX belongs to an optional extra;
the root package never imports the integration.

### Verified result

The complete suite passes with 141 tests. Focused ONNX tests cover generated-ID
and independent-node-order stability, localized RMSNorm diffs, operation counts,
visible constant and dimension changes, scope recovery, bounded tensor decoding,
model preservation, inferred file output, replacement, and native error propagation.
A subprocess test confirms core imports with ONNX and Lightning unavailable.

Ruff linting and formatting, strict Pyrefly, locked environment synchronization,
and the portable `0.10.0` wheel build pass. The executable example writes a UTF-8
RMSNormalization summary with dynamic dimensions and no temporary node counters.

### Limitations

This is an inspection summary, not an equivalence checker. Internal rewiring can
leave it unchanged. Symbolic dimension names and equality constraints are omitted.
Source scope recovery depends on surviving tensor names and exporter metadata;
fallback grouping is heuristic. No performance or cross-exporter stability
guarantee is claimed.
