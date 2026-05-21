from __future__ import annotations

import argparse
import base64
import copy
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

from .onnx_verify import compare_tensors
from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values


DTYPES = {
    "<f2": np.dtype("<f2"),
    "<f4": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "fp16": np.dtype("<f2"),
    "fp32": np.dtype("<f4"),
}


def parse_shape(value: str) -> list[int]:
    parts = re.split(r"[,xX]", value)
    if not parts or any(part.strip() == "" for part in parts):
        raise ValueError(f"Invalid input shape: {value!r}")

    shape: list[int] = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid input shape dimension: {part!r}") from exc
        if dim <= 0:
            raise ValueError(f"Input shape dimensions must be positive: {value!r}")
        shape.append(dim)
    return shape


def dtype_from_name(name: str) -> np.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported input dtype: {name}") from exc


def load_input_array(
    *,
    image: str | Path | None,
    input_bin: str | Path | None,
    input_shape: str | None,
    input_dtype: str,
    image_size: int,
) -> np.ndarray:
    if image and input_bin:
        raise ValueError("--image and --input-bin are mutually exclusive")
    if input_bin:
        if not input_shape:
            raise ValueError("--input-shape is required with --input-bin")
        shape = parse_shape(input_shape)
        dtype = dtype_from_name(input_dtype)
        array = np.fromfile(input_bin, dtype=dtype)
        expected_size = int(np.prod(shape))
        if array.size != expected_size:
            raise ValueError(
                f"Input bin element count mismatch: got {array.size}, expected {expected_size} for shape {shape}"
            )
        return array.reshape(shape).astype("float32")
    if not image:
        raise ValueError("Either --image or --input-bin is required")
    return load_static_pixel_values(image, image_size).numpy().astype("float32")


def select_node_outputs(
    model: onnx.ModelProto,
    *,
    op_types: set[str] | None = None,
    name_regex: str | None = None,
) -> list[str]:
    pattern = re.compile(name_regex) if name_regex else None
    selected: list[str] = []
    for node in model.graph.node:
        if op_types and node.op_type not in op_types:
            continue
        haystack = " ".join([node.name, node.op_type, *node.output])
        if pattern and not pattern.search(haystack):
            continue
        selected.extend(output for output in node.output if output)
    return selected


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    infos = {}
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            infos[value.name] = value
    return infos


def augment_graph_outputs(model: onnx.ModelProto, tensor_names: Iterable[str]) -> onnx.ModelProto:
    augmented = copy.deepcopy(model)
    try:
        inferred = shape_inference.infer_shapes(augmented)
        info_map = value_info_by_name(inferred)
    except Exception:
        info_map = value_info_by_name(augmented)

    existing = {output.name for output in augmented.graph.output}
    for name in dict.fromkeys(tensor_names):
        if name in existing:
            continue
        if name in info_map:
            augmented.graph.output.append(copy.deepcopy(info_map[name]))
        else:
            augmented.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
        existing.add(name)
    return augmented


def save_augmented_model(
    input_onnx: str | Path,
    output_onnx: str | Path,
    *,
    tensor_names: list[str] | None = None,
    op_types: set[str] | None = None,
    name_regex: str | None = None,
    max_outputs: int | None = None,
) -> dict[str, object]:
    model = onnx.load(input_onnx)
    selected = list(tensor_names or [])
    if op_types or name_regex:
        selected.extend(select_node_outputs(model, op_types=op_types, name_regex=name_regex))
    selected = list(dict.fromkeys(selected))
    if max_outputs is not None:
        selected = selected[:max_outputs]
    if not selected:
        raise SystemExit("No intermediate tensors selected")

    augmented = augment_graph_outputs(model, selected)
    output_path = Path(output_onnx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(augmented, output_path)
    return {
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_path),
        "added_output_count": len(selected),
        "added_outputs": selected,
        "total_output_count": len(augmented.graph.output),
    }


def safe_npz_key(name: str) -> str:
    encoded = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")
    return f"tensor_{encoded}"


def save_outputs_npz(output_npz: str | Path, outputs: dict[str, np.ndarray]) -> dict[str, object]:
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_to_name = {safe_npz_key(name): name for name in outputs}
    np.savez(output_path, **{key: outputs[name] for key, name in key_to_name.items()})
    manifest = {
        "npz": str(output_path),
        "tensor_count": len(outputs),
        "output_names": list(outputs),
        "key_to_name": key_to_name,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_npz_named(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    manifest_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    if not manifest_path.exists():
        return arrays
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key_to_name = manifest.get("key_to_name", {})
    return {key_to_name.get(key, key): value for key, value in arrays.items()}


def dump_onnx_outputs(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort

    pixel_values = load_input_array(
        image=args.image,
        input_bin=args.input_bin,
        input_shape=args.input_shape,
        input_dtype=args.input_dtype,
        image_size=args.image_size,
    )
    session = ort.InferenceSession(args.onnx, providers=[args.provider])
    input_name = args.input_name or session.get_inputs()[0].name
    output_names = args.output_name or [output.name for output in session.get_outputs()]
    output_values = session.run(output_names, {input_name: pixel_values})
    outputs = {name: value for name, value in zip(output_names, output_values)}
    manifest = save_outputs_npz(args.output_npz, outputs)
    result = {
        "onnx": args.onnx,
        "image": args.image,
        "input_bin": args.input_bin,
        "input_shape": list(pixel_values.shape),
        "input_name": input_name,
        "output_names": output_names,
        "output_npz": args.output_npz,
        "output_count": len(outputs),
        "manifest": manifest,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def compare_npz_outputs(left_npz: str | Path, right_npz: str | Path) -> dict[str, object]:
    left = load_npz_named(left_npz)
    right = load_npz_named(right_npz)
    left_names = set(left)
    right_names = set(right)
    common = sorted(left_names & right_names)
    records = []
    for name in common:
        try:
            metrics = compare_tensors(np.asarray(right[name]), np.asarray(left[name]))
            records.append({"name": name, "metrics": metrics})
        except ValueError as exc:
            records.append({"name": name, "error": str(exc)})
    records.sort(
        key=lambda record: (
            record.get("metrics", {}).get("cosine", -1.0) if "metrics" in record else -1.0,
            -record.get("metrics", {}).get("mean_abs_diff", 0.0) if "metrics" in record else 0.0,
        )
    )
    return {
        "left_npz": str(left_npz),
        "right_npz": str(right_npz),
        "common_count": len(common),
        "left_only": sorted(left_names - right_names),
        "right_only": sorted(right_names - left_names),
        "records": records,
    }


def parse_op_types(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(part for part in value.split(",") if part)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX intermediate output debug utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List candidate node outputs.")
    list_parser.add_argument("--onnx", required=True)
    list_parser.add_argument("--op-type", action="append")
    list_parser.add_argument("--name-regex")
    list_parser.add_argument("--max-outputs", type=int)

    augment_parser = subparsers.add_parser("augment", help="Add selected tensors to graph outputs.")
    augment_parser.add_argument("--onnx", required=True)
    augment_parser.add_argument("--output-onnx", required=True)
    augment_parser.add_argument("--tensor-name", action="append")
    augment_parser.add_argument("--tensor-file", help="Text file with one tensor name per line.")
    augment_parser.add_argument("--op-type", action="append")
    augment_parser.add_argument("--name-regex")
    augment_parser.add_argument("--max-outputs", type=int)
    augment_parser.add_argument("--output-json")

    dump_parser = subparsers.add_parser("dump", help="Run a debug ONNX and save all requested outputs to NPZ.")
    dump_parser.add_argument("--onnx", required=True)
    dump_parser.add_argument("--image")
    dump_parser.add_argument("--input-bin")
    dump_parser.add_argument("--input-shape")
    dump_parser.add_argument("--input-dtype", default="float32", choices=sorted(DTYPES))
    dump_parser.add_argument("--output-npz", required=True)
    dump_parser.add_argument("--input-name")
    dump_parser.add_argument("--output-name", action="append")
    dump_parser.add_argument("--image-size", type=int, default=STATIC_IMAGE_SIZE)
    dump_parser.add_argument("--provider", default="CPUExecutionProvider")
    dump_parser.add_argument("--output-json")

    compare_parser = subparsers.add_parser("compare", help="Compare two NPZ dumps by common tensor names.")
    compare_parser.add_argument("--left-npz", required=True)
    compare_parser.add_argument("--right-npz", required=True)
    compare_parser.add_argument("--output-json")

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "list":
        model = onnx.load(args.onnx)
        selected = select_node_outputs(
            model,
            op_types=parse_op_types(args.op_type),
            name_regex=args.name_regex,
        )
        if args.max_outputs is not None:
            selected = selected[: args.max_outputs]
        return {"onnx": args.onnx, "count": len(selected), "outputs": selected}

    if args.command == "augment":
        tensor_names = list(args.tensor_name or [])
        if args.tensor_file:
            tensor_names.extend(
                line.strip()
                for line in Path(args.tensor_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
        result = save_augmented_model(
            args.onnx,
            args.output_onnx,
            tensor_names=tensor_names,
            op_types=parse_op_types(args.op_type),
            name_regex=args.name_regex,
            max_outputs=args.max_outputs,
        )
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    if args.command == "dump":
        return dump_onnx_outputs(args)

    if args.command == "compare":
        result = compare_npz_outputs(args.left_npz, args.right_npz)
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    raise SystemExit(f"Unknown command: {args.command}")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
