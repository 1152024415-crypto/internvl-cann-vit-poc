from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .onnx_pair_eval import discover_images
from .onnx_verify import compare_tensors, summarize_tensor
from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

INPUT_SHAPE = [1, 3, 448, 448]
OUTPUT_SHAPE = [1, 256, 1024]
DEFAULT_OM_OUTPUT_SUFFIX = "_visual_tokens_om_fp32.bin"
DEFAULT_ONNX_OUTPUT_SUFFIX = "_visual_tokens_int8_onnx_fp32.bin"
DEFAULT_INPUT_SUFFIX = "_pixel_values_fp32.bin"

DTYPES = {
    "float32": np.dtype("<f4"),
    "fp32": np.dtype("<f4"),
    "<f4": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "fp16": np.dtype("<f2"),
    "<f2": np.dtype("<f2"),
}


def build_case_id(relative_path: str | Path) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    raw = "_".join(parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    return safe or "case"


def make_unique_case_ids(relative_paths: Iterable[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    result: dict[Path, str] = {}
    for path in relative_paths:
        base = build_case_id(path)
        count = counts.get(base, 0)
        counts[base] = count + 1
        result[path] = base if count == 0 else f"{base}_{count + 1}"
    return result


def dtype_from_name(name: str) -> np.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Supported: {', '.join(sorted(DTYPES))}") from exc


def write_fp32_bin(path: str | Path, tensor: np.ndarray) -> dict[str, object]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tensor, dtype="<f4")
    array.tofile(output_path)
    return {
        "path": str(output_path),
        "shape": [int(dim) for dim in array.shape],
        "dtype": "float32",
        "byte_count": int(output_path.stat().st_size),
    }


def read_bin_tensor(path: str | Path, *, shape: list[int], dtype: str = "float32") -> np.ndarray:
    tensor_path = Path(path)
    array = np.fromfile(tensor_path, dtype=dtype_from_name(dtype)).astype(np.float32, copy=False)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"{tensor_path} has {array.size} elements, expected {expected} for shape {shape}")
    return array.reshape(shape)


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def load_om_output_map(path: str | Path | None, *, base_dir: Path) -> dict[str, Path]:
    if path is None:
        return {}
    map_path = Path(path)
    result: dict[str, Path] = {}
    with map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = True
        if has_header:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(reader.fieldnames) < 2:
                raise ValueError(f"OM output map must have case_id and om_output_bin columns: {map_path}")
            case_field = "case_id" if "case_id" in reader.fieldnames else reader.fieldnames[0]
            output_field = "om_output_bin" if "om_output_bin" in reader.fieldnames else reader.fieldnames[1]
            for row in reader:
                result[row[case_field]] = _resolve_path(row[output_field], base_dir)
            return result

        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                result[row[0]] = _resolve_path(row[1], base_dir)
    return result


def save_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_compare_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "image",
        "status",
        "cosine",
        "mean_abs_diff",
        "max_abs_diff",
        "onnx_output_bin",
        "om_output_bin",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            compare = record.get("compare", {})
            writer.writerow(
                {
                    "case_id": record.get("case_id"),
                    "image": record.get("image"),
                    "status": record.get("status"),
                    "cosine": compare.get("cosine") if isinstance(compare, dict) else None,
                    "mean_abs_diff": compare.get("mean_abs_diff") if isinstance(compare, dict) else None,
                    "max_abs_diff": compare.get("max_abs_diff") if isinstance(compare, dict) else None,
                    "onnx_output_bin": record.get("onnx_output_bin"),
                    "om_output_bin": record.get("om_output_bin"),
                    "error": record.get("error"),
                }
            )


def save_run_plan_csv(path: str | Path, cases: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "image", "input_bin", "expected_om_output_bin", "onnx_output_bin"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field) for field in fields})


def prepare_onnx_reference(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort

    work_dir = Path(args.work_dir)
    input_dir = Path(args.input_dir) if args.input_dir else work_dir / "inputs"
    onnx_output_dir = Path(args.onnx_output_dir) if args.onnx_output_dir else work_dir / "int8_onnx_outputs"
    om_output_dir = Path(args.om_output_dir) if args.om_output_dir else work_dir / "om_outputs"
    manifest_path = Path(args.manifest) if args.manifest else work_dir / "manifest.json"
    run_plan_path = Path(args.run_plan_csv) if args.run_plan_csv else work_dir / "om_run_plan.csv"

    image_root = Path(args.image_dir)
    images = discover_images(image_root)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found under {image_root}")

    relative_paths = [image.relative_to(image_root) for image in images]
    case_ids = make_unique_case_ids(relative_paths)

    session = ort.InferenceSession(args.onnx, providers=[args.provider])
    input_name = args.input_name or session.get_inputs()[0].name
    output_name = args.output_name or session.get_outputs()[0].name

    cases: list[dict[str, object]] = []
    for index, image in enumerate(images, start=1):
        relative = image.relative_to(image_root)
        case_id = case_ids[relative]
        pixel_values = load_static_pixel_values(image, args.image_size).numpy().astype("<f4")
        onnx_output = session.run([output_name], {input_name: pixel_values})[0].astype("<f4")

        input_bin = input_dir / f"{case_id}{DEFAULT_INPUT_SUFFIX}"
        onnx_output_bin = onnx_output_dir / f"{case_id}{DEFAULT_ONNX_OUTPUT_SUFFIX}"
        expected_om_output_bin = om_output_dir / f"{case_id}{args.om_output_suffix}"

        input_info = write_fp32_bin(input_bin, pixel_values)
        output_info = write_fp32_bin(onnx_output_bin, onnx_output)
        case = {
            "index": index,
            "case_id": case_id,
            "image": relative.as_posix(),
            "input_bin": str(input_bin),
            "onnx_output_bin": str(onnx_output_bin),
            "expected_om_output_bin": str(expected_om_output_bin),
            "input_shape": input_info["shape"],
            "output_shape": output_info["shape"],
            "input_dtype": "float32",
            "onnx_output_dtype": "float32",
        }
        cases.append(case)
        if args.progress_every and index % args.progress_every == 0:
            print(json.dumps({"processed": index, "latest": relative.as_posix()}, ensure_ascii=False), flush=True)

    result: dict[str, object] = {
        "kind": "int8_onnx_reference_for_om_compare",
        "onnx": args.onnx,
        "provider": args.provider,
        "image_dir": str(image_root),
        "input_name": input_name,
        "output_name": output_name,
        "work_dir": str(work_dir),
        "input_dir": str(input_dir),
        "onnx_output_dir": str(onnx_output_dir),
        "om_output_dir": str(om_output_dir),
        "om_output_suffix": args.om_output_suffix,
        "count": len(cases),
        "cases": cases,
    }
    save_json(manifest_path, result)
    save_run_plan_csv(run_plan_path, cases)
    result["manifest"] = str(manifest_path)
    result["run_plan_csv"] = str(run_plan_path)
    return result


def summarize_compare_records(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [record for record in records if record.get("status") == "ok" and isinstance(record.get("compare"), dict)]
    missing = [record for record in records if record.get("status") == "missing_om_output"]
    failed = [record for record in records if record.get("status") == "compare_failed"]
    summary: dict[str, object] = {
        "count": len(records),
        "valid_count": len(valid),
        "missing_count": len(missing),
        "failed_count": len(failed),
    }
    if not valid:
        summary.update(
            {
                "cosine_min": None,
                "cosine_mean": None,
                "cosine_p01": None,
                "mean_abs_diff_mean": None,
                "mean_abs_diff_max": None,
                "max_abs_diff_max": None,
            }
        )
        return summary

    cosines = [float(record["compare"]["cosine"]) for record in valid]  # type: ignore[index]
    mean_abs = [float(record["compare"]["mean_abs_diff"]) for record in valid]  # type: ignore[index]
    max_abs = [float(record["compare"]["max_abs_diff"]) for record in valid]  # type: ignore[index]
    summary.update(
        {
            "cosine_min": float(np.min(cosines)),
            "cosine_mean": float(np.mean(cosines)),
            "cosine_p01": float(np.quantile(cosines, 0.01)),
            "mean_abs_diff_mean": float(np.mean(mean_abs)),
            "mean_abs_diff_max": float(np.max(mean_abs)),
            "max_abs_diff_max": float(np.max(max_abs)),
        }
    )
    return summary


def compare_manifest_outputs(
    *,
    manifest_path: str | Path,
    om_output_dir: str | Path,
    om_output_suffix: str = DEFAULT_OM_OUTPUT_SUFFIX,
    om_output_map: str | Path | None = None,
    om_output_dtype: str = "float32",
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base_dir = manifest_file.parent
    om_dir = Path(om_output_dir)
    output_map = load_om_output_map(om_output_map, base_dir=base_dir)

    records: list[dict[str, object]] = []
    for case in manifest.get("cases", []):
        case_id = case["case_id"]
        shape = [int(dim) for dim in case.get("output_shape", OUTPUT_SHAPE)]
        onnx_output_bin = _resolve_path(case["onnx_output_bin"], base_dir)
        om_output_bin = output_map.get(case_id, om_dir / f"{case_id}{om_output_suffix}")
        record: dict[str, object] = {
            "case_id": case_id,
            "image": case.get("image"),
            "onnx_output_bin": str(onnx_output_bin),
            "om_output_bin": str(om_output_bin),
        }
        if not om_output_bin.exists():
            record["status"] = "missing_om_output"
            record["error"] = f"Missing OM output: {om_output_bin}"
            records.append(record)
            continue
        try:
            reference = read_bin_tensor(onnx_output_bin, shape=shape, dtype="float32")
            actual = read_bin_tensor(om_output_bin, shape=shape, dtype=om_output_dtype)
            record["status"] = "ok"
            record["compare"] = compare_tensors(actual, reference)
            record["onnx_output"] = summarize_tensor(reference)
            record["om_output"] = summarize_tensor(actual)
        except Exception as exc:
            record["status"] = "compare_failed"
            record["error"] = str(exc)
        records.append(record)

    return {
        "manifest": str(manifest_file),
        "om_output_dir": str(om_dir),
        "om_output_dtype": om_output_dtype,
        "om_output_suffix": om_output_suffix,
        "summary": summarize_compare_records(records),
        "records": records,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and compare INT8 ONNX references against OM output bins.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate pixel_values bins and INT8 ONNX reference outputs.")
    prepare_parser.add_argument("--onnx", required=True, help="INT8 ONNX path.")
    prepare_parser.add_argument("--image-dir", required=True, help="Image directory to preprocess recursively.")
    prepare_parser.add_argument("--work-dir", required=True, help="Directory for manifest, input bins, and reference bins.")
    prepare_parser.add_argument("--input-dir", help="Defaults to <work-dir>/inputs.")
    prepare_parser.add_argument("--onnx-output-dir", help="Defaults to <work-dir>/int8_onnx_outputs.")
    prepare_parser.add_argument("--om-output-dir", help="Defaults to <work-dir>/om_outputs.")
    prepare_parser.add_argument("--manifest", help="Defaults to <work-dir>/manifest.json.")
    prepare_parser.add_argument("--run-plan-csv", help="Defaults to <work-dir>/om_run_plan.csv.")
    prepare_parser.add_argument("--input-name")
    prepare_parser.add_argument("--output-name")
    prepare_parser.add_argument("--image-size", type=int, default=STATIC_IMAGE_SIZE)
    prepare_parser.add_argument("--provider", default="CPUExecutionProvider")
    prepare_parser.add_argument("--max-images", type=int)
    prepare_parser.add_argument("--progress-every", type=int, default=25)
    prepare_parser.add_argument("--om-output-suffix", default=DEFAULT_OM_OUTPUT_SUFFIX)

    compare_parser = subparsers.add_parser("compare", help="Compare OM output bins against INT8 ONNX reference bins.")
    compare_parser.add_argument("--manifest", required=True)
    compare_parser.add_argument("--om-output-dir", required=True)
    compare_parser.add_argument("--om-output-suffix", default=DEFAULT_OM_OUTPUT_SUFFIX)
    compare_parser.add_argument("--om-output-map", help="CSV: case_id,om_output_bin. Relative paths resolve from manifest dir.")
    compare_parser.add_argument("--om-output-dtype", default="float32", choices=sorted(DTYPES))
    compare_parser.add_argument("--output-json")
    compare_parser.add_argument("--output-csv")

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "prepare":
        return prepare_onnx_reference(args)

    if args.command == "compare":
        result = compare_manifest_outputs(
            manifest_path=args.manifest,
            om_output_dir=args.om_output_dir,
            om_output_suffix=args.om_output_suffix,
            om_output_map=args.om_output_map,
            om_output_dtype=args.om_output_dtype,
        )
        if args.output_json:
            save_json(args.output_json, result)
        if args.output_csv:
            save_compare_csv(args.output_csv, result["records"])  # type: ignore[arg-type]
        return result

    raise SystemExit(f"Unknown command: {args.command}")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
