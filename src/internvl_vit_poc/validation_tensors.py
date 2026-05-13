from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

INPUT_SHAPE = [1, 3, 448, 448]
OUTPUT_SHAPE = [1, 256, 1024]
MODEL_ARTIFACT = "internvl3_5_vit_projector_fp32_opset18_staticpos.om"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fp32_bin(path: str | Path, tensor: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tensor, dtype="<f4")
    array.tofile(path)
    return {
        "path": str(path),
        "shape": [int(dim) for dim in array.shape],
        "dtype": "float32",
        "byte_count": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_reference_tensor(path: str | Path) -> np.ndarray:
    import torch

    loaded = torch.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        loaded = loaded.detach().cpu().numpy()
    return np.asarray(loaded, dtype=np.float32)


def build_case_metadata(
    *,
    case_name: str,
    source_image: str,
    model_artifact: str,
    baseline_source: str,
    input_info: dict[str, Any],
    output_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image": source_image,
        "model_artifact": model_artifact,
        "baseline_source": baseline_source,
        "input": input_info,
        "expected_output": output_info,
        "thresholds": {
            "cosine_min": 0.999,
            "mean_abs_diff_max": 0.01,
        },
    }


def save_case_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def export_case(
    *,
    case_name: str,
    image_path: str | Path,
    expected_output_path: str | Path,
    output_dir: str | Path,
    model_artifact: str = MODEL_ARTIFACT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pixel_values = load_static_pixel_values(image_path, STATIC_IMAGE_SIZE).numpy().astype(np.float32)
    expected_output = load_reference_tensor(expected_output_path)

    if list(pixel_values.shape) != INPUT_SHAPE:
        raise ValueError(f"input shape mismatch: got {list(pixel_values.shape)}, expected {INPUT_SHAPE}")
    if list(expected_output.shape) != OUTPUT_SHAPE:
        raise ValueError(f"output shape mismatch: got {list(expected_output.shape)}, expected {OUTPUT_SHAPE}")

    input_info = write_fp32_bin(output_dir / f"{case_name}_pixel_values_fp32.bin", pixel_values)
    output_info = write_fp32_bin(output_dir / f"{case_name}_visual_tokens_fp32.bin", expected_output)
    metadata = build_case_metadata(
        case_name=case_name,
        source_image=str(image_path),
        model_artifact=model_artifact,
        baseline_source=str(expected_output_path),
        input_info=input_info,
        output_info=output_info,
    )
    save_case_metadata(output_dir / f"{case_name}.metadata.json", metadata)
    return metadata


def parse_case(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("case must be name=image_path,baseline_output_path")
    paths = parts[1].split(",", 1)
    if len(paths) != 2 or not paths[0] or not paths[1]:
        raise argparse.ArgumentTypeError("case must be name=image_path,baseline_output_path")
    return parts[0], paths[0], paths[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw fp32 validation tensors for HarmonyOS OM tests.")
    parser.add_argument("--output-dir", required=True, help="Directory for .bin and metadata outputs.")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        help="Test case in the form name=image_path,baseline_output.pt",
    )
    parser.add_argument("--model-artifact", default=MODEL_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        export_case(
            case_name=name,
            image_path=image_path,
            expected_output_path=expected_output_path,
            output_dir=args.output_dir,
            model_artifact=args.model_artifact,
        )
        for name, image_path, expected_output_path in args.case
    ]
    print(json.dumps({"cases": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
