from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from .baseline import DEFAULT_MODEL_ID, PINNED_REVISION, load_model, resolve_dtype
from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

DEFAULT_OPSET = 18
INPUT_NAME = "pixel_values"
STATIC_POSITION_EMBEDDING_METADATA_KEY = "static_position_embedding"


def _shape_to_list(shape: Iterable[int]) -> list[int]:
    return [int(dim) for dim in shape]


class VisionLastHiddenStateWrapper(torch.nn.Module):
    """ONNX-friendly wrapper that returns only ViT last_hidden_state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state


class ProjectorVisualTokensWrapper(torch.nn.Module):
    """ONNX-friendly wrapper for ViT + pixel_shuffle + mlp1."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.extract_feature(pixel_values)


def patch_static_position_embedding(model) -> bool:
    """Bypass bicubic position embedding interpolation for fixed-size CANN export.

    InternVL always calls ``F.interpolate(..., mode="bicubic")`` in the vision
    embedding path, even when the input image is already the model's native
    size. CANN Kit's OMG converter rejects that ONNX Resize mode. For the first
    PoC we only export ``1 x 3 x 448 x 448``, so the patch grid already matches
    the learned position embedding grid and interpolation is an identity.
    """

    embeddings = getattr(getattr(model, "vision_model", None), "embeddings", None)
    if embeddings is None or not hasattr(embeddings, "_get_pos_embed"):
        return False

    image_size = int(getattr(embeddings, "image_size"))
    patch_size = int(getattr(embeddings, "patch_size"))
    native_grid = image_size // patch_size
    original_get_pos_embed = embeddings._get_pos_embed

    def fixed_get_pos_embed(pos_embed, height, width):
        if int(height) == native_grid and int(width) == native_grid:
            return pos_embed
        return original_get_pos_embed(pos_embed, height, width)

    embeddings._get_pos_embed = fixed_get_pos_embed
    return True


def make_wrapper(model, target: str):
    if target == "vision":
        return VisionLastHiddenStateWrapper(model), "last_hidden_state"
    if target == "projector":
        return ProjectorVisualTokensWrapper(model), "visual_tokens"
    raise ValueError(f"Unsupported ONNX target: {target}")


def make_onnx_metadata(
    *,
    model_id: str,
    revision: str,
    image_path: str | Path,
    output_path: str | Path,
    target: str,
    dtype: str,
    device: str,
    opset: int,
    input_shape: Iterable[int],
    output_shape: Iterable[int],
    input_name: str,
    output_name: str,
    static_position_embedding: bool = False,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "revision": revision,
        "image_path": str(image_path),
        "output_path": str(output_path),
        "target": target,
        "dtype": dtype,
        "device": device,
        "opset": int(opset),
        "input_name": input_name,
        "output_name": output_name,
        "input_shape": _shape_to_list(input_shape),
        "output_shape": _shape_to_list(output_shape),
        "dynamic_axes": False,
        STATIC_POSITION_EMBEDDING_METADATA_KEY: bool(static_position_embedding),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def export_onnx(
    wrapper,
    pixel_values,
    output_path: str | Path,
    *,
    opset: int,
    output_name: str,
) -> Path:
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pixel_values,),
            str(output_path),
            input_names=[INPUT_NAME],
            output_names=[output_name],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export InternVL ViT or visual projector path to static ONNX."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--target",
        choices=("vision", "projector"),
        default="vision",
        help="vision exports last_hidden_state; projector exports visual tokens.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required for InternVL custom modeling files.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use only files already present in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--static-position-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bypass bicubic position embedding interpolation when exporting the "
            "fixed native 448 image size used by this PoC."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    model = load_model(
        args.model_id,
        args.dtype,
        args.trust_remote_code,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    static_position_embedding = False
    if args.static_position_embedding:
        static_position_embedding = patch_static_position_embedding(model)
    wrapper, output_name = make_wrapper(model, args.target)

    torch_dtype = resolve_dtype(args.dtype)
    pixel_values = load_static_pixel_values(args.image, STATIC_IMAGE_SIZE).to(
        device=args.device,
        dtype=torch_dtype,
    )

    with torch.no_grad():
        output = wrapper(pixel_values)

    output_path = export_onnx(
        wrapper,
        pixel_values,
        args.output,
        opset=args.opset,
        output_name=output_name,
    )

    metadata = make_onnx_metadata(
        model_id=args.model_id,
        revision=args.revision,
        image_path=args.image,
        output_path=output_path,
        target=args.target,
        dtype=args.dtype,
        device=args.device,
        opset=args.opset,
        input_shape=tuple(pixel_values.shape),
        output_shape=tuple(output.shape),
        input_name=INPUT_NAME,
        output_name=output_name,
        static_position_embedding=static_position_embedding,
    )

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
