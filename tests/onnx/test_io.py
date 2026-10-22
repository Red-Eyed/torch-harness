"""Exercise real shape inference and summary-file boundary behavior."""

from pathlib import Path

import onnx
import pytest
from onnx import ModelProto, TensorProto, helper

from torch_harness.onnx import write_text_graph


@pytest.fixture
def relu_model() -> ModelProto:
    """Provide a model whose output shape needs inference."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])],
        "example",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


def test_write_infers_shapes_without_mutating_model(
    relu_model: ModelProto, tmp_path: Path
) -> None:
    """Create nested directories, replace old text, and preserve caller state."""
    before = relu_model.SerializeToString()
    destination = tmp_path / "nested" / "model.onnx.txt"
    scopes = {"output": "activation"}
    assert (
        write_text_graph(relu_model, destination, source_scopes=scopes) == destination
    )
    text = destination.read_text(encoding="utf-8")
    assert "activation:" in text
    assert "Relu() : (float[2, 8]) -> (float[2, 8])" in text
    destination.write_text("old summary", encoding="utf-8")
    write_text_graph(relu_model, destination, source_scopes=scopes)
    assert destination.read_text(encoding="utf-8") == text
    assert relu_model.SerializeToString() == before


def test_write_propagates_filesystem_failure(
    relu_model: ModelProto, tmp_path: Path
) -> None:
    """Writing to a directory should retain the native filesystem exception."""
    with pytest.raises(IsADirectoryError):
        write_text_graph(relu_model, tmp_path)


def test_write_propagates_inference_failure(
    relu_model: ModelProto, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference errors should propagate before creating the destination tree."""
    failure = onnx.shape_inference.InferenceError("cannot infer shape")

    def reject_inference(model: ModelProto) -> ModelProto:
        """Raise the original inference failure at the third-party boundary."""
        raise failure

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", reject_inference)
    destination = tmp_path / "absent" / "model.txt"
    with pytest.raises(onnx.shape_inference.InferenceError) as raised:
        write_text_graph(relu_model, destination)
    assert raised.value is failure
    assert not destination.parent.exists()
