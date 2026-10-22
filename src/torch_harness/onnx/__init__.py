"""Human-readable ONNX summaries; requires the ``onnx`` installation extra."""

from torch_harness.onnx._io import write_text_graph
from torch_harness.onnx._text_graph import collect_source_scopes, format_text_graph

__all__ = ["collect_source_scopes", "format_text_graph", "write_text_graph"]
