from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values


def summarize_tensor(tensor: np.ndarray) -> dict[str, object]:
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "mean": float(np.mean(tensor)),
        "std": float(np.std(tensor)),
    }


def compare_tensors(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: actual={actual.shape}, reference={reference.shape}")

    diff = np.abs(actual - reference)
    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    cosine = float(
        np.dot(actual_flat, reference_flat)
        / (np.linalg.norm(actual_flat) * np.linalg.norm(reference_flat))
    )
    return {
        "shape": [int(dim) for dim in actual.shape],
        "reference_shape": [int(dim) for dim in reference.shape],
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "cosine": cosine,
    }


def save_tensor(path: str | Path, tensor: np.ndarray) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, tensor)
        return
    torch.save(torch.from_numpy(tensor), path)


def load_reference_tensor(path: str | Path) -> np.ndarray:
    import torch

    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    loaded = torch.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        loaded = loaded.detach().cpu().numpy()
    return np.asarray(loaded)


def run(args: argparse.Namespace) -> dict[str, object]:
    import onnxruntime as ort

    pixel_values = load_static_pixel_values(args.image, args.image_size).numpy().astype("float32")
    session = ort.InferenceSession(args.onnx, providers=[args.provider])
    output_name = args.output_name or session.get_outputs()[0].name
    output = session.run([output_name], {"pixel_values": pixel_values})[0]

    result: dict[str, object] = {
        "onnx_path": args.onnx,
        "image_path": args.image,
        "input_shape": [int(dim) for dim in pixel_values.shape],
        "output_name": output_name,
        "output": summarize_tensor(output),
    }

    if args.save_output:
        save_tensor(args.save_output, output)
        result["saved_output"] = args.save_output

    if args.compare:
        reference = load_reference_tensor(args.compare)
        result["compare_path"] = args.compare
        result["compare"] = compare_tensors(output, reference)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ONNX visual model on one image.")
    parser.add_argument("--onnx", required=True, help="Path to the ONNX model.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--output-name", default=None, help="ONNX output name. Defaults to the first output.")
    parser.add_argument("--save-output", default=None, help="Optional .pt or .npy tensor output path.")
    parser.add_argument("--compare", default=None, help="Optional .pt or .npy reference tensor.")
    parser.add_argument("--image-size", type=int, default=STATIC_IMAGE_SIZE)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
