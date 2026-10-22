# ONNX module summaries

Install the optional integration with `uv pip install 'torch-harness[onnx]'`.
The core package does not import or require ONNX, and these utilities do not
require ONNX Runtime or an exporter backend.

```python
from pathlib import Path

import onnx

from torch_harness.onnx import write_text_graph

model = onnx.load("model.onnx")
write_text_graph(model, Path("artifacts/model.onnx.txt"))
```

`write_text_graph` infers shapes on a copy, creates parent directories, and
replaces the destination with UTF-8 text. It returns the supplied path. Shape
inference and filesystem exceptions propagate unchanged; neither the input
model nor its binary file is modified. Load external tensor data with the model
before calling it.

For a string without file I/O or shape inference, use `format_text_graph(model)`.

## What appears in a diff

Source modules and operation signatures are sorted. Repeated identical
signatures are counted rather than assigned node IDs. For example:

```text
norm:
  inputs: input, weight
  weight: float[256]
  operations:
    RMSNormalization(axis=-1, epsilon=1e-06) : (float[1, 256], float[256]) -> (float[1, 256])
```

The decomposed implementation lists `Pow`, `ReduceMean`, `Add`, `Sqrt`, and
the remaining actual operations in this same module. A fusion therefore
changes that module's operation signatures without shifting numbered aliases
throughout the file. Different shapes or attributes produce separate signatures.

Tensor dimensions come from the graph's recorded types, including inferred
types when using the writer. `?` means a dynamic dimension; `unknown` means
missing type information. Scalars have shape `[]`, and tensors with unknown
rank are marked explicitly. Generated symbolic dimension names are omitted.

## Retaining source module paths

PyTorch exports may include node `namespace` metadata identifying source
modules. Optimization can remove that metadata. Capture it from the exported
model and pass it when writing the optimized model:

```python
from torch_harness.onnx import collect_source_scopes, write_text_graph

scopes = collect_source_scopes(exported_model)
# Pass the captured mapping when summarizing the optimizer's output.
write_text_graph(
    optimized_model, Path("artifacts/model.onnx.txt"), source_scopes=scopes
)
```

This mapping associates output tensor names with source module paths. It works
when optimization preserves those tensor names. Without usable metadata, named
weight/bias inputs identify modules. Single-consumer normalization chains can
inherit their scale parameter's module; other unassigned operations stay in
`graph`. Source module paths retain meaningful architecture indices such as
`layers.0`; temporary node indices are absent.

## Boundaries

- This is a module-level inspection summary, not a reloadable graph or a proof
  of model equivalence. Internal tensor wiring, execution order, and symbolic
  dimension equality constraints are omitted. Rewiring can leave it unchanged.
- Conventional `weight`/`bias` names identify parameters. Other small
  initializers are treated as structural constants. Their values are shown;
  large tensor payloads and named parameter values are omitted.
- Grouping without exporter metadata is heuristic. Operators in different
  modules remain distinct; identical signatures within one module are counted.
- Tensor names in public graph inputs/outputs and source module paths are
  retained. Renaming those public boundaries produces a diff.
- The formatter never runs inference or mutates its input. The writer runs ONNX
  shape inference on a copy, which can fail or leave dimensions unknown for
  operators without shape inference support. It does not execute the model.
