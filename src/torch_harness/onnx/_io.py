"""Write shape-aware ONNX inspection summaries without changing inference graphs."""

from collections.abc import Mapping
from pathlib import Path

import onnx
from onnx import ModelProto

from torch_harness.onnx._text_graph import format_text_graph


def write_text_graph(
    model: ModelProto,
    output_path: Path,
    *,
    source_scopes: Mapping[str, str] | None = None,
) -> Path:
    """Infer shapes on a copy and write a UTF-8 module summary.

    Creates parent directories and replaces an existing text file. The input
    model and its inference artifact are unchanged. This does not validate
    numerical equivalence or execute the model.

    Args:
        model: Loaded ONNX model, including any external tensor data it needs.
        output_path: Destination for the inspection summary.
        source_scopes: Tensor-to-module mapping captured before optimization
            removed source metadata, if available.

    Returns:
        The output path supplied by the caller.

    Raises:
        onnx.shape_inference.InferenceError: If ONNX shape inference fails.
        OSError: If creating directories or writing the summary fails.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    summary = format_text_graph(inferred, source_scopes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    return output_path
