from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .om_onnx_eval import read_bin_tensor, save_json
from .onnx_intermediate_debug import load_npz_named
from .onnx_verify import compare_tensors, summarize_tensor


def candidate_element_count(byte_count: int, element_size: int) -> int | None:
    if byte_count % element_size != 0:
        return None
    return byte_count // element_size


def inventory_dump_files(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"OMC dump directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"OMC dump path is not a directory: {root_path}")

    files = sorted(path for path in root_path.rglob("*") if path.is_file())
    records = []
    for path in files:
        byte_count = path.stat().st_size
        records.append(
            {
                "relative_path": path.relative_to(root_path).as_posix(),
                "path": str(path),
                "byte_count": byte_count,
                "fp32_element_count": candidate_element_count(byte_count, 4),
                "fp16_element_count": candidate_element_count(byte_count, 2),
            }
        )
    return {
        "root": str(root_path),
        "count": len(records),
        "records": records,
    }


def save_inventory_csv(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "path",
        "byte_count",
        "fp32_element_count",
        "fp16_element_count",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_output_names(
    *,
    onnx_npz: str | Path,
    output_names_json: str | Path | None = None,
    output_names_file: str | Path | None = None,
) -> list[str]:
    if output_names_json and output_names_file:
        raise ValueError("Use either --output-names-json or --output-names-file, not both")

    if output_names_json:
        payload = json.loads(Path(output_names_json).read_text(encoding="utf-8"))
        names = payload.get("output_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{output_names_json} does not contain string list field output_names")
        return list(names)

    if output_names_file:
        return [
            line.strip()
            for line in Path(output_names_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    manifest_path = Path(onnx_npz).with_suffix(Path(onnx_npz).suffix + ".manifest.json")
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = payload.get("output_names")
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return list(names)

    return list(load_npz_named(onnx_npz))


def summarize_compare_records(records: list[dict[str, object]]) -> dict[str, object]:
    ok = [record for record in records if record.get("status") == "ok" and isinstance(record.get("compare"), dict)]
    errors = [record for record in records if record.get("status") != "ok"]
    summary: dict[str, object] = {
        "count": len(records),
        "ok_count": len(ok),
        "error_count": len(errors),
    }
    if ok:
        cosines = [float(record["compare"]["cosine"]) for record in ok]  # type: ignore[index]
        mean_abs = [float(record["compare"]["mean_abs_diff"]) for record in ok]  # type: ignore[index]
        max_abs = [float(record["compare"]["max_abs_diff"]) for record in ok]  # type: ignore[index]
        summary.update(
            {
                "cosine_min": float(np.min(cosines)),
                "cosine_mean": float(np.mean(cosines)),
                "mean_abs_diff_mean": float(np.mean(mean_abs)),
                "max_abs_diff_max": float(np.max(max_abs)),
            }
        )
    else:
        summary.update(
            {
                "cosine_min": None,
                "cosine_mean": None,
                "mean_abs_diff_mean": None,
                "max_abs_diff_max": None,
            }
        )
    return summary


def first_bad_record(records: list[dict[str, object]], cosine_threshold: float) -> dict[str, object] | None:
    for record in records:
        if record.get("status") != "ok":
            return record
        compare = record.get("compare", {})
        if isinstance(compare, dict) and float(compare.get("cosine", 1.0)) < cosine_threshold:
            return record
    return None


def compare_indexed_outputs(
    *,
    onnx_npz: str | Path,
    om_output_dir: str | Path,
    output_names_json: str | Path | None = None,
    output_names_file: str | Path | None = None,
    om_output_prefix: str = "output_",
    om_output_dtype: str = "float32",
    cosine_threshold: float = 0.99,
) -> dict[str, object]:
    onnx_outputs = load_npz_named(onnx_npz)
    output_names = load_output_names(
        onnx_npz=onnx_npz,
        output_names_json=output_names_json,
        output_names_file=output_names_file,
    )
    om_dir = Path(om_output_dir)
    records: list[dict[str, object]] = []

    for index, name in enumerate(output_names):
        om_path = om_dir / f"{om_output_prefix}{index}"
        record: dict[str, object] = {
            "index": index,
            "onnx_name": name,
            "om_output": str(om_path),
        }
        try:
            if name not in onnx_outputs:
                raise KeyError(f"ONNX tensor not found: {name}")
            if not om_path.exists():
                raise FileNotFoundError(f"OMC output not found: {om_path}")
            reference = np.asarray(onnx_outputs[name], dtype=np.float32)
            actual = read_bin_tensor(om_path, shape=[int(dim) for dim in reference.shape], dtype=om_output_dtype)
            record["status"] = "ok"
            record["shape"] = [int(dim) for dim in reference.shape]
            record["dtype"] = om_output_dtype
            record["compare"] = compare_tensors(actual, reference)
            record["onnx_summary"] = summarize_tensor(reference)
            record["om_summary"] = summarize_tensor(actual)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        records.append(record)

    first_bad = first_bad_record(records, cosine_threshold)
    return {
        "onnx_npz": str(onnx_npz),
        "om_output_dir": str(om_dir),
        "om_output_prefix": om_output_prefix,
        "om_output_dtype": om_output_dtype,
        "cosine_threshold": cosine_threshold,
        "output_names": output_names,
        "summary": summarize_compare_records(records),
        "first_bad": first_bad,
        "records": records,
    }


def save_compare_csv(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "onnx_name",
        "om_output",
        "shape",
        "dtype",
        "status",
        "cosine",
        "mean_abs_diff",
        "max_abs_diff",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            compare = record.get("compare", {})
            writer.writerow(
                {
                    "index": record.get("index"),
                    "onnx_name": record.get("onnx_name"),
                    "om_output": record.get("om_output"),
                    "shape": ",".join(str(dim) for dim in record.get("shape", [])),
                    "dtype": record.get("dtype"),
                    "status": record.get("status"),
                    "cosine": compare.get("cosine") if isinstance(compare, dict) else None,
                    "mean_abs_diff": compare.get("mean_abs_diff") if isinstance(compare, dict) else None,
                    "max_abs_diff": compare.get("max_abs_diff") if isinstance(compare, dict) else None,
                    "error": record.get("error"),
                }
            )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OMC dump alignment utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Inventory files in a pulled OMC dump directory.")
    inventory_parser.add_argument("--dump-dir", required=True)
    inventory_parser.add_argument("--output-json")
    inventory_parser.add_argument("--output-csv")

    indexed_parser = subparsers.add_parser(
        "compare-indexed",
        help="Compare debug ONNX outputs against model_run_tool output_0/output_1 files.",
    )
    indexed_parser.add_argument("--onnx-npz", required=True)
    indexed_parser.add_argument("--om-output-dir", required=True)
    indexed_parser.add_argument("--output-names-json", help="JSON from dump_onnx_outputs.py containing output_names.")
    indexed_parser.add_argument("--output-names-file", help="Text file with one ONNX output name per line.")
    indexed_parser.add_argument("--om-output-prefix", default="output_")
    indexed_parser.add_argument("--om-output-dtype", default="float32", choices=["<f2", "<f4", "float16", "float32", "fp16", "fp32"])
    indexed_parser.add_argument("--cosine-threshold", type=float, default=0.99)
    indexed_parser.add_argument("--output-json")
    indexed_parser.add_argument("--output-csv")

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "inventory":
        result = inventory_dump_files(args.dump_dir)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.output_csv:
            save_inventory_csv(args.output_csv, result["records"])
        return result

    if args.command == "compare-indexed":
        result = compare_indexed_outputs(
            onnx_npz=args.onnx_npz,
            om_output_dir=args.om_output_dir,
            output_names_json=args.output_names_json,
            output_names_file=args.output_names_file,
            om_output_prefix=args.om_output_prefix,
            om_output_dtype=args.om_output_dtype,
            cosine_threshold=args.cosine_threshold,
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
