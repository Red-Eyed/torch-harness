"""Render ONNX structure for diffs without generated IDs or weight payloads."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod

import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto


def collect_source_scopes(model: ModelProto) -> dict[str, str]:
    """Capture PyTorch source modules before optimization drops metadata.

    Args:
        model: Exported ONNX model with PyTorch namespace metadata, if present.

    Returns:
        Output tensor names mapped to source modules. Nodes without usable
        namespace metadata are omitted. The model is never modified.
    """
    scopes: dict[str, str] = {}
    for node in model.graph.node:
        metadata = {entry.key: entry.value for entry in node.metadata_props}
        namespace = metadata.get("namespace", "").split("/")
        if len(namespace) < 2:
            continue
        scope = namespace[-2].partition(":")[0].strip()
        if scope and not scope.startswith("_empty_nn_module_stack"):
            scopes.update((name, scope) for name in node.output if name)
    return scopes


def _parameter_scope(node: NodeProto) -> str:
    """Recover a module path from surviving parameter names."""
    for name in node.input:
        if name.endswith((".weight", ".bias")):
            return name.rsplit(".", 1)[0]
    return ""


def _node_scopes(graph: GraphProto, hints: Mapping[str, str]) -> list[str]:
    """Prefer source metadata, recovering unannotated norm chains from their scale."""
    scopes = [
        next((hints[name] for name in node.output if name in hints), "")
        for node in graph.node
    ]
    producers = {
        name: index for index, node in enumerate(graph.node) for name in node.output
    }
    consumers = Counter(name for node in graph.node for name in node.input)
    norm_ops = {"Pow", "Mul", "ReduceMean", "Add", "Sqrt", "Div", "Reciprocal", "Cast"}
    for index, node in enumerate(graph.node):
        scopes[index] = scopes[index] or _parameter_scope(node)
    for index, node in enumerate(graph.node):
        if not scopes[index] or node.op_type != "Mul" or not _parameter_scope(node):
            continue
        pending = list(node.input)
        while pending:
            name = pending.pop()
            upstream = producers.get(name)
            if upstream is None or scopes[upstream] or consumers[name] != 1:
                continue
            producer = graph.node[upstream]
            if producer.op_type not in norm_ops:
                continue
            scopes[upstream] = scopes[index]
            pending.extend(producer.input)
    return [scope or "graph" for scope in scopes]


def _tensor_type(tensor: TensorProto) -> str:
    """Describe an initializer's dtype and dimensions without its values."""
    dtype = TensorProto.DataType.Name(tensor.data_type).lower()
    return f"{dtype}[{', '.join(str(size) for size in tensor.dims)}]"


def _small_tensor(tensor: TensorProto) -> str:
    """Inline small structural constants without materializing large tensors."""
    if prod(tensor.dims) > 8 or tensor.data_type == TensorProto.STRING:
        return _tensor_type(tensor)
    values = onnx.numpy_helper.to_array(tensor)
    if values.ndim == 0 and values.dtype.kind == "f":
        return f"{float(values.item()):.7g}"
    return str(values.tolist())


def _attribute(attribute: AttributeProto) -> str:
    """Render scalar and structural attributes with bounded tensor payloads."""
    match attribute.type:
        case AttributeProto.FLOAT:
            return f"{attribute.f:.7g}"
        case AttributeProto.INT:
            return str(attribute.i)
        case AttributeProto.STRING:
            return repr(attribute.s.decode("utf-8", errors="replace"))
        case AttributeProto.INTS:
            return str(list(attribute.ints))
        case AttributeProto.FLOATS:
            return "[" + ", ".join(f"{value:.7g}" for value in attribute.floats) + "]"
        case AttributeProto.TENSOR:
            return _small_tensor(attribute.t)
        case AttributeProto.GRAPH:
            return "<subgraph>"
        case AttributeProto.GRAPHS:
            return f"<{len(attribute.graphs)} subgraphs>"
        case _:
            return str(onnx.helper.printable_attribute(attribute))


def _constant(node: NodeProto) -> str:
    """Inline an ONNX Constant's value rather than assigning it an unstable ID."""
    return ", ".join(_attribute(attribute) for attribute in node.attribute)


def _is_parameter(name: str) -> bool:
    """Distinguish conventional learned parameter names from structural constants."""
    return name in {"weight", "bias"} or name.endswith((".weight", ".bias"))


def _input_role(node: NodeProto, position: int, opsets: Mapping[str, int]) -> str:
    """Name a constant operand by its schema role rather than its generated ID."""
    if node.op_type == "Pow" and position == 1:
        return "exponent"
    try:
        schema = onnx.defs.get_schema(
            node.op_type, opsets.get(node.domain, 1), node.domain
        )
    except onnx.defs.SchemaError:
        return "constant"
    if position >= len(schema.inputs):
        return schema.inputs[-1].name.lower() if schema.inputs else "constant"
    return schema.inputs[position].name.lower()


def _proto_name(value: AttributeProto | onnx.ValueInfoProto) -> str:
    """Read a protobuf name through a typed boundary for deterministic sorting."""
    return value.name


def _operation_signature(
    node: NodeProto,
    constants: Mapping[str, str],
    opsets: Mapping[str, int],
    tensor_types: Mapping[str, str],
) -> str:
    """Describe an actual operation using attributes and small constant operands."""
    arguments = [
        f"{_input_role(node, position, opsets)}={constants[name]}"
        for position, name in enumerate(node.input)
        if name in constants
    ]
    attributes: list[AttributeProto] = list(node.attribute)
    attributes.sort(key=_proto_name)
    arguments.extend(f"{attr.name}={_attribute(attr)}" for attr in attributes)
    operator = f"{node.domain}::{node.op_type}" if node.domain else node.op_type
    signature = f"{operator}({', '.join(arguments)})"
    inputs = ", ".join(
        tensor_types.get(name, "unknown") if name else "absent" for name in node.input
    )
    outputs = ", ".join(
        tensor_types.get(name, "unknown") for name in node.output if name
    )
    signature += f" : ({inputs}) -> ({outputs})"
    for attr in attributes:
        graphs = [attr.g] if attr.type == AttributeProto.GRAPH else list(attr.graphs)
        nested = Counter(
            format_text_graph(onnx.helper.make_model(graph)) for graph in graphs
        )
        for summary, count in sorted(nested.items()):
            signature += f"\n  {attr.name}{_multiplicity(count)}:\n"
            signature += "\n".join(f"    {line}" for line in summary.splitlines())
    return signature


def _constant_values(graph: GraphProto) -> dict[str, str]:
    """Inline constants without decoding weights or large initializer payloads."""
    values = {
        tensor.name: _small_tensor(tensor)
        for tensor in graph.initializer
        if not _is_parameter(tensor.name) and prod(tensor.dims) <= 8
    }
    for node in graph.node:
        if node.op_type == "Constant":
            values.update((name, _constant(node)) for name in node.output if name)
    return values


@dataclass
class _ModuleSummary:
    """Collect module dependencies, parameter shapes, and an operation multiset."""

    inputs: set[str] = field(default_factory=set)
    parameters: set[str] = field(default_factory=set)
    operations: Counter[str] = field(default_factory=Counter)


def _module_summaries(
    graph: GraphProto,
    scopes: list[str],
    constants: Mapping[str, str],
    opsets: Mapping[str, int],
    tensor_types: Mapping[str, str],
) -> dict[str, _ModuleSummary]:
    """Aggregate structural facts independently of node order and temporary names."""
    groups: dict[str, _ModuleSummary] = defaultdict(_ModuleSummary)
    owners = {
        name: scope
        for node, scope in zip(graph.node, scopes, strict=True)
        for name in node.output
    }
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    for node, scope in zip(graph.node, scopes, strict=True):
        if node.op_type == "Constant":
            continue
        group = groups[scope]
        group.operations[
            _operation_signature(node, constants, opsets, tensor_types)
        ] += 1
        for name in node.input:
            if not name or name in constants:
                continue
            if name in initializers:
                tensor_type = _tensor_type(initializers[name])
                label = (
                    name.removeprefix(f"{scope}.")
                    if _is_parameter(name)
                    else f"tensor<{tensor_type}>"
                )
                group.inputs.add(label)
                group.parameters.add(f"{label}: {tensor_type}")
                continue
            source = owners.get(name, name)
            if source != scope:
                group.inputs.add(source)
    return groups


def _multiplicity(count: int) -> str:
    """Mark repeated identical signatures without assigning node identities."""
    return f" \N{MULTIPLICATION SIGN} {count}" if count > 1 else ""


def _type_description(type_proto: onnx.TypeProto) -> str:
    """Describe tensor or sequence dimensions without generated symbolic names."""
    if type_proto.HasField("sequence_type"):
        return f"sequence<{_type_description(type_proto.sequence_type.elem_type)}>"
    tensor = type_proto.tensor_type
    if not tensor.elem_type:
        return "unknown"
    dtype = TensorProto.DataType.Name(tensor.elem_type).lower()
    if not tensor.HasField("shape"):
        return f"{dtype}[unknown rank]"
    dimensions = [
        str(dim.dim_value) if dim.HasField("dim_value") else "?"
        for dim in tensor.shape.dim
    ]
    return f"{dtype}[{', '.join(dimensions)}]"


def _tensor_types(graph: GraphProto) -> dict[str, str]:
    """Collect recorded dimensions without running inference or decoding tensors."""
    types = {
        value.name: _type_description(value.type)
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    types.update((tensor.name, _tensor_type(tensor)) for tensor in graph.initializer)
    for node in graph.node:
        for attr in node.attribute:
            if (
                node.op_type == "Constant"
                and attr.name == "value"
                and attr.type == AttributeProto.TENSOR
            ):
                types.update(
                    (name, _tensor_type(attr.t)) for name in node.output if name
                )
    return types


def format_text_graph(
    model: ModelProto, source_scopes: Mapping[str, str] | None = None
) -> str:
    """Summarize module dependencies and operation signatures without node identities.

    Modules and signatures are sorted; identical signatures are counted. This
    deliberately omits internal tensor wiring and execution order, so rewiring
    within a module may leave the summary unchanged. Source metadata or parameter
    paths identify modules; unassigned operations stay in ``graph``. This is an
    inspection summary, not a reloadable ONNX model. Signatures include recorded
    input/output shapes; ``?`` means a dynamic dimension and ``unknown`` means
    missing type information. Shape inference is not run by this formatter.

    Args:
        model: Loaded ONNX model whose recorded types and operations to inspect.
        source_scopes: Tensor-to-module associations captured before optimization
            removed source metadata. These override metadata in the model.

    Returns:
        A deterministic text summary ending in a newline. Neither the model nor
        its tensor payloads are modified. Internal wiring is not represented.
    """
    graph = model.graph
    hints = collect_source_scopes(model)
    hints.update(source_scopes or {})
    scopes = _node_scopes(graph, hints)
    constants = _constant_values(graph)
    opsets = {entry.domain: entry.version for entry in model.opset_import}
    tensor_types = _tensor_types(graph)
    groups = _module_summaries(graph, scopes, constants, opsets, tensor_types)
    lines = [
        "# ONNX module summary; internal wiring and weight payloads omitted",
        "",
        "inputs:",
    ]
    inputs: list[onnx.ValueInfoProto] = list(graph.input)
    outputs: list[onnx.ValueInfoProto] = list(graph.output)
    lines.extend(
        f"  {value.name}: {tensor_types[value.name]}"
        for value in sorted(inputs, key=_proto_name)
    )
    for scope, group in sorted(groups.items()):
        lines.extend(
            [
                "",
                f"{scope}:",
                f"  inputs: {', '.join(sorted(group.inputs)) or '(none)'}",
            ]
        )
        lines.extend(f"  {parameter}" for parameter in sorted(group.parameters))
        lines.append("  operations:")
        for signature, count in sorted(group.operations.items()):
            lines.append(f"    {signature}{_multiplicity(count)}")
    owners = {
        name: scope
        for node, scope in zip(graph.node, scopes, strict=True)
        for name in node.output
    }
    lines.extend(["", "outputs:"])
    lines.extend(
        f"  {value.name}: {tensor_types[value.name]} "
        f"from {owners.get(value.name, value.name)}"
        for value in sorted(outputs, key=_proto_name)
    )
    return "\n".join(lines) + "\n"
