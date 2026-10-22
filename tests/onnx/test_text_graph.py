"""Verify that structural diffs isolate normalization implementation changes."""

from difflib import unified_diff

import onnx
import pytest
from onnx import ModelProto, TensorProto, helper

from torch_harness.onnx import collect_source_scopes, format_text_graph


def _normalization_model(*, fused: bool, prefix: str = "generated") -> ModelProto:
    """Build equivalent normalization variants followed by an unchanged consumer."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])
    y = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 8])
    weight = helper.make_tensor("block.norm.weight", TensorProto.FLOAT, [8], [1.0] * 8)
    output = f"{prefix}_norm"
    if fused:
        nodes = [
            helper.make_node(
                "RMSNormalization",
                ["input", weight.name],
                [output],
                epsilon=1e-6,
                axis=-1,
            )
        ]
        initializers = [weight]
    else:
        initializers = [
            weight,
            helper.make_tensor(f"{prefix}_two", TensorProto.FLOAT, [], [2.0]),
            helper.make_tensor(f"{prefix}_epsilon", TensorProto.FLOAT, [], [1e-6]),
            helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [-1]),
        ]
        nodes = [
            helper.make_node(
                "Pow", ["input", initializers[1].name], [f"{prefix}_square"]
            ),
            helper.make_node(
                "ReduceMean",
                [f"{prefix}_square", initializers[3].name],
                [f"{prefix}_mean"],
                keepdims=1,
            ),
            helper.make_node(
                "Add", [f"{prefix}_mean", initializers[2].name], [f"{prefix}_add"]
            ),
            helper.make_node("Sqrt", [f"{prefix}_add"], [f"{prefix}_sqrt"]),
            helper.make_node(
                "Div", ["input", f"{prefix}_sqrt"], [f"{prefix}_normalized"]
            ),
            helper.make_node("Mul", [f"{prefix}_normalized", weight.name], [output]),
        ]
    nodes.append(helper.make_node("Relu", [output], ["result"]))
    for index, node in enumerate(nodes):
        node.name = f"{prefix}_node_{index}"
    graph = helper.make_graph(nodes, prefix, [x], [y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    return onnx.shape_inference.infer_shapes(model)


@pytest.mark.parametrize("fused", [False, True])
def test_generated_identifiers_do_not_change_summary(fused: bool) -> None:
    """Node, graph, intermediate, and constant IDs must not create diff noise."""
    first = _normalization_model(fused=fused, prefix="first_42")
    second = _normalization_model(fused=fused, prefix="second_917")
    assert format_text_graph(first) == format_text_graph(second)


def test_rmsnorm_diff_preserves_consumer_summary() -> None:
    """Fusion should change only normalization signatures, preserving dependencies."""
    decomposed = format_text_graph(_normalization_model(fused=False))
    fused = format_text_graph(_normalization_model(fused=True))
    changes = [
        line
        for line in unified_diff(decomposed.splitlines(), fused.splitlines())
        if line[:1] in {"+", "-"}
    ]
    # Two file headers, six primitives removed, one fused node added.
    assert len(changes) == 9
    assert (
        "+    RMSNormalization(axis=-1, epsilon=1e-06) : "
        "(float[1, 8], float[8]) -> (float[1, 8])" in changes
    )
    consumer = "graph:\n  inputs: block.norm\n  operations:\n    Relu()"
    assert consumer in decomposed
    assert consumer in fused
    assert "RMSNormalization" not in decomposed


def test_captured_scopes_survive_metadata_removal() -> None:
    """Original module paths should still group the runtime-optimized graph."""
    model = _normalization_model(fused=True)
    node = model.graph.node[-1]
    node.metadata_props.add(
        key="namespace", value=": Model/block.activation: ReLU/relu: aten.relu.default"
    )
    scopes = collect_source_scopes(model)
    node.ClearField("metadata_props")
    assert (
        "block.activation:\n  inputs: block.norm\n  operations:\n    Relu()"
        in format_text_graph(model, scopes)
    )


def test_large_constants_and_weights_are_not_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summaries must not expand tensor payloads even for Constant attributes."""
    model = _normalization_model(fused=True)
    tensor = helper.make_tensor("large", TensorProto.FLOAT, [100], [99.0] * 100)
    model.graph.node.insert(
        0, helper.make_node("Constant", [], ["large_value"], value=tensor)
    )

    def reject_decode(tensor: TensorProto, base_dir: str = "") -> None:
        """Fail if rendering attempts to decode an omitted tensor payload."""
        raise AssertionError("Tensor payload decoded")

    monkeypatch.setattr(onnx.numpy_helper, "to_array", reject_decode)
    summary = format_text_graph(model)
    assert "99.0" not in summary
    assert "weight" in summary


def test_rendering_preserves_model() -> None:
    """Inspection must leave the inference model and its parameters unchanged."""
    model = _normalization_model(fused=False)
    before = model.SerializeToString()
    assert format_text_graph(model) == format_text_graph(model)
    assert model.SerializeToString() == before


def test_independent_node_order_does_not_change_summary() -> None:
    """Reordering independent nodes must not introduce new names or reorder lines."""
    first = _normalization_model(fused=True)
    branch = helper.make_node("Relu", ["input"], ["unrelated_generated_93"])
    first.graph.node.insert(0, branch)
    first.graph.value_info.append(
        helper.make_tensor_value_info(branch.output[0], TensorProto.FLOAT, [1, 8])
    )
    second = _normalization_model(fused=True)
    second.graph.node.append(branch)
    second.graph.value_info.append(
        helper.make_tensor_value_info(branch.output[0], TensorProto.FLOAT, [1, 8])
    )
    assert format_text_graph(first) == format_text_graph(second)
    repeated = "Relu() : (float[1, 8]) -> (float[1, 8]) \N{MULTIPLICATION SIGN} 2"
    assert repeated in format_text_graph(first)


def test_adding_operation_changes_only_its_count() -> None:
    """An extra independent operation should affect one signature's multiplicity."""
    model = _normalization_model(fused=True)
    before = format_text_graph(model)
    model.graph.node.insert(
        0, helper.make_node("Relu", ["input"], ["another_generated_5"])
    )
    model.graph.value_info.append(
        helper.make_tensor_value_info("another_generated_5", TensorProto.FLOAT, [1, 8])
    )
    after = format_text_graph(model)
    changes = [
        line
        for line in unified_diff(before.splitlines(), after.splitlines())
        if line[:1] in {"+", "-"}
    ]
    signature = "Relu() : (float[1, 8]) -> (float[1, 8])"
    assert changes[2:] == [
        "-  inputs: block.norm",
        "+  inputs: block.norm, input",
        f"-    {signature}",
        f"+    {signature} \N{MULTIPLICATION SIGN} 2",
    ]
    assert "output_" not in after
    assert "generated" not in after


def test_changed_structural_constant_is_visible() -> None:
    """Hiding generated IDs must not hide an epsilon change in decomposed RMSNorm."""
    model = _normalization_model(fused=False)
    before = format_text_graph(model)
    model.graph.initializer[2].float_data[0] = 2e-6
    after = format_text_graph(model)
    assert "Add(b=1e-06)" in before
    assert "Add(b=2e-06)" in after


def test_shape_changes_are_visible_without_symbolic_ids() -> None:
    """Changing a recorded dimension must affect the operation signature."""
    model = _normalization_model(fused=True)
    before = format_text_graph(model)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 4
    after = format_text_graph(model)
    assert "(float[1, 8], float[8])" in before
    assert "(float[4, 8], float[8])" in after
    dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    dimension.dim_param = "generated_827"
    dynamic = format_text_graph(model)
    dimension.dim_param = "generated_929"
    assert format_text_graph(model) == dynamic
    assert "(float[?, 8], float[8])" in dynamic
