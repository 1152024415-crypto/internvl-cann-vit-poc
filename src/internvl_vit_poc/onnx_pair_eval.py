from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .onnx_verify import compare_tensors, summarize_tensor
from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".png", ".webp", ".tiff", ".tif"}


def normalize_key(path: str | Path) -> str:
    return Path(path).as_posix().replace("\\", "/")


def discover_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_labels_csv(path: str | Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return labels
            if len(reader.fieldnames) < 2:
                raise ValueError(f"labels CSV must have at least two columns: {path}")
            image_field = "image" if "image" in reader.fieldnames else reader.fieldnames[0]
            label_field = "label" if "label" in reader.fieldnames else reader.fieldnames[1]
            for row in reader:
                labels[normalize_key(row[image_field])] = int(row[label_field])
            return labels

        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 2:
                continue
            labels[normalize_key(row[0])] = int(row[1])
    return labels


def lookup_label(labels: dict[str, int], image_path: Path, image_root: Path | None = None) -> int | None:
    candidates = [normalize_key(image_path), image_path.name]
    if image_root is not None:
        try:
            candidates.insert(0, normalize_key(image_path.relative_to(image_root)))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate in labels:
            return labels[candidate]
    return None


def top1_prediction(output: np.ndarray) -> int:
    tensor = np.asarray(output)
    if tensor.ndim >= 2 and tensor.shape[0] == 1:
        tensor = tensor.reshape(-1)
    return int(np.argmax(tensor))


def summarize_pair_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "count": 0,
            "label_count": 0,
            "fp32_accuracy": None,
            "int8_accuracy": None,
            "same_pred_rate": None,
        }

    label_records = [record for record in records if record.get("label") is not None]
    pred_records = [record for record in records if record.get("same_pred") is not None]
    same_pred_count = sum(1 for record in pred_records if record.get("same_pred"))
    cosines = [float(record["compare"]["cosine"]) for record in records]  # type: ignore[index]
    mean_abs = [float(record["compare"]["mean_abs_diff"]) for record in records]  # type: ignore[index]
    max_abs = [float(record["compare"]["max_abs_diff"]) for record in records]  # type: ignore[index]

    summary: dict[str, object] = {
        "count": len(records),
        "label_count": len(label_records),
        "same_pred_rate": same_pred_count / len(pred_records) if pred_records else None,
        "cosine_min": float(np.min(cosines)),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_p01": float(np.quantile(cosines, 0.01)),
        "mean_abs_diff_mean": float(np.mean(mean_abs)),
        "mean_abs_diff_max": float(np.max(mean_abs)),
        "max_abs_diff_max": float(np.max(max_abs)),
    }

    if label_records:
        fp32_correct = sum(
            1
            for record in label_records
            if record.get("fp32_correct", record.get("fp32_pred") == record.get("label"))
        )
        int8_correct = sum(
            1
            for record in label_records
            if record.get("int8_correct", record.get("int8_pred") == record.get("label"))
        )
        summary["fp32_accuracy"] = fp32_correct / len(label_records)
        summary["int8_accuracy"] = int8_correct / len(label_records)
    else:
        summary["fp32_accuracy"] = None
        summary["int8_accuracy"] = None

    return summary


def save_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "label",
        "fp32_pred",
        "int8_pred",
        "fp32_correct",
        "int8_correct",
        "same_pred",
        "cosine",
        "mean_abs_diff",
        "max_abs_diff",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            compare = record["compare"]  # type: ignore[index]
            writer.writerow(
                {
                    "image": record["image"],
                    "label": record.get("label"),
                    "fp32_pred": record.get("fp32_pred"),
                    "int8_pred": record.get("int8_pred"),
                    "fp32_correct": record.get("fp32_correct"),
                    "int8_correct": record.get("int8_correct"),
                    "same_pred": record.get("same_pred"),
                    "cosine": compare["cosine"],  # type: ignore[index]
                    "mean_abs_diff": compare["mean_abs_diff"],  # type: ignore[index]
                    "max_abs_diff": compare["max_abs_diff"],  # type: ignore[index]
                }
            )


def run_pair_evaluation(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort

    image_root = Path(args.image_dir)
    images = discover_images(image_root)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found under {image_root}")

    labels = load_labels_csv(args.labels_csv) if args.labels_csv else {}

    fp32_session = ort.InferenceSession(args.fp32_onnx, providers=[args.provider])
    int8_session = ort.InferenceSession(args.int8_onnx, providers=[args.provider])
    fp32_input_name = args.input_name or fp32_session.get_inputs()[0].name
    int8_input_name = args.input_name or int8_session.get_inputs()[0].name
    fp32_output_name = args.fp32_output_name or fp32_session.get_outputs()[0].name
    int8_output_name = args.int8_output_name or int8_session.get_outputs()[0].name

    records: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        pixel_values = load_static_pixel_values(image_path, args.image_size).numpy().astype("float32")
        fp32_output = fp32_session.run([fp32_output_name], {fp32_input_name: pixel_values})[0]
        int8_output = int8_session.run([int8_output_name], {int8_input_name: pixel_values})[0]
        compare = compare_tensors(np.asarray(int8_output), np.asarray(fp32_output))

        label = lookup_label(labels, image_path, image_root) if labels else None
        fp32_pred = top1_prediction(fp32_output) if args.compute_top1 or label is not None else None
        int8_pred = top1_prediction(int8_output) if args.compute_top1 or label is not None else None

        record: dict[str, object] = {
            "index": index,
            "image": normalize_key(image_path.relative_to(image_root)),
            "compare": compare,
            "fp32_output": summarize_tensor(np.asarray(fp32_output)),
            "int8_output": summarize_tensor(np.asarray(int8_output)),
        }
        if fp32_pred is not None and int8_pred is not None:
            record["fp32_pred"] = fp32_pred
            record["int8_pred"] = int8_pred
            record["same_pred"] = fp32_pred == int8_pred
        if label is not None:
            record["label"] = label
            record["fp32_correct"] = fp32_pred == label
            record["int8_correct"] = int8_pred == label
        records.append(record)

        if args.progress_every and index % args.progress_every == 0:
            print(json.dumps({"processed": index, "latest": record["image"]}, ensure_ascii=False), flush=True)

    result: dict[str, object] = {
        "fp32_onnx": args.fp32_onnx,
        "int8_onnx": args.int8_onnx,
        "image_dir": str(image_root),
        "fp32_input_name": fp32_input_name,
        "int8_input_name": int8_input_name,
        "fp32_output_name": fp32_output_name,
        "int8_output_name": int8_output_name,
        "summary": summarize_pair_records(records),
        "records": records,
    }
    if args.output_json:
        save_json(args.output_json, result)
    if args.output_csv:
        save_csv(args.output_csv, records)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare FP32 and INT8 ONNX models on one image set.")
    parser.add_argument("--fp32-onnx", required=True)
    parser.add_argument("--int8-onnx", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--labels-csv", default=None, help="CSV with image,label columns. Optional for embedding models.")
    parser.add_argument("--input-name", default=None, help="Input tensor name. Defaults to each model's first input.")
    parser.add_argument("--fp32-output-name", default=None, help="Defaults to FP32 model's first output.")
    parser.add_argument("--int8-output-name", default=None, help="Defaults to INT8 model's first output.")
    parser.add_argument("--image-size", type=int, default=STATIC_IMAGE_SIZE)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--compute-top1", action="store_true", help="Compute argmax predictions without labels.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run_pair_evaluation(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
