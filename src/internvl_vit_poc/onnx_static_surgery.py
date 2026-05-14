from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


CLASS_EMBEDDING_INITIALIZER = "model.vision_model.embeddings.class_embedding"
EMBEDDING_CONCAT_OUTPUT = "/vision_model/embeddings/Concat_1_output_0"


def replace_class_embedding_expand(model: onnx.ModelProto) -> bool:
    """Replace the fixed batch=1 class-token Expand chain with its initializer.

    PyTorch exports ``class_embedding.expand(batch, -1, -1)`` as a small
    Equal/Where/Expand/Cast shape subgraph. For this PoC the input batch is
    statically 1, and the class embedding initializer is already shaped
    ``[1, 1, hidden_size]``. Feeding that initializer directly to the embedding
    concat keeps the numerics identical while avoiding CANN-unsupported shape
    helper ops in the front of the graph.
    """

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    if CLASS_EMBEDDING_INITIALIZER not in initializer_names:
        return False

    for node in model.graph.node:
        if node.op_type != "Concat" or EMBEDDING_CONCAT_OUTPUT not in node.output:
            continue
        if not node.input:
            return False
        node.input[0] = CLASS_EMBEDDING_INITIALIZER
        return True

    return False


def prune_unused_graph_inputs(model: onnx.ModelProto) -> int:
    """Remove nodes and initializers that are no longer reachable from outputs."""

    producers: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            producers[output] = index

    required_values = {output.name for output in model.graph.output}
    required_nodes: set[int] = set()
    stack = list(required_values)

    while stack:
        value = stack.pop()
        producer_index = producers.get(value)
        if producer_index is None or producer_index in required_nodes:
            continue

        required_nodes.add(producer_index)
        producer = model.graph.node[producer_index]
        for input_name in producer.input:
            if input_name and input_name not in required_values:
                required_values.add(input_name)
                stack.append(input_name)

    old_node_count = len(model.graph.node)
    kept_nodes = [
        node for index, node in enumerate(model.graph.node) if index in required_nodes
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in required_values
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    return old_node_count - len(kept_nodes)


def simplify_for_cann(input_path: str | Path, output_path: str | Path) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    model = onnx.load(input_path)

    replaced = replace_class_embedding_expand(model)
    removed_nodes = prune_unused_graph_inputs(model) if replaced else 0

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "static_class_embedding": replaced,
        "removed_nodes": removed_nodes,
    }


def update_metadata(metadata_path: str | Path, result: dict[str, object]) -> None:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_path"] = str(result["output_path"])
    metadata["static_class_embedding"] = bool(result["static_class_embedding"])
    metadata["cann_graph_surgery"] = ["static_class_embedding"]
    metadata["cann_graph_surgery_removed_nodes"] = int(result["removed_nodes"])
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply fixed-shape CANN graph surgery to an InternVL ONNX file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata",
        help="Optional ONNX metadata JSON to update after graph surgery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = simplify_for_cann(args.input, args.output)
    if args.metadata:
        update_metadata(args.metadata, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
