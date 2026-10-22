"""Save an inspectable ONNX RMSNorm summary without trained weights."""

from pathlib import Path

from loguru import logger
from onnx import TensorProto, helper

from torch_harness.onnx import write_text_graph


def main() -> None:
    """Build a minimal ONNX graph and write its summary beside this example."""
    weight = helper.make_tensor("norm.weight", TensorProto.FLOAT, [8], [1.0] * 8)
    graph = helper.make_graph(
        [
            helper.make_node(
                "RMSNormalization",
                ["input", weight.name],
                ["output"],
                axis=-1,
                epsilon=1e-6,
            )
        ],
        "normalization",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 8])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    destination = write_text_graph(model, Path(__file__).with_suffix(".txt"))
    logger.info("Wrote ONNX module summary to {}", destination)


if __name__ == "__main__":
    main()
