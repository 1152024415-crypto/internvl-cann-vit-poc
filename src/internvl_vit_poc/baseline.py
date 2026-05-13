from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metadata import make_baseline_metadata
from .model_split import disable_flash_attention, save_vision_config, save_vision_state
from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3_5-1B-Instruct"
PINNED_REVISION = "5648fa26ff23acaba53588936d9f1dfaf305f522"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split InternVL ViT weights and record a fixed-shape PyTorch baseline."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=PINNED_REVISION,
        help="Pinned Hugging Face commit used for model files and remote code.",
    )
    parser.add_argument("--image", required=True, help="Image used for the baseline tensor.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and baseline artifacts.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or other torch device string.")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="Torch dtype for model inference.",
    )
    parser.add_argument(
        "--include-projector",
        action="store_true",
        help="Run and save the visual path after InternVL pixel_shuffle + mlp1.",
    )
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
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    import torch

    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_model(
    model_id: str,
    dtype_name: str,
    trust_remote_code: bool,
    *,
    revision: str,
    local_files_only: bool = False,
):
    from transformers import AutoModel

    torch_dtype = resolve_dtype(dtype_name)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=revision,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    )
    disable_flash_attention(model)
    model.eval()
    return model


def run_visual_baseline(model, pixel_values, *, include_projector: bool):
    import torch

    with torch.no_grad():
        if include_projector:
            output = model.extract_feature(pixel_values)
            output_kind = "projected_visual_tokens"
        else:
            output = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            output_kind = "vision_last_hidden_state"
    return output, output_kind


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        args.model_id,
        args.dtype,
        args.trust_remote_code,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)

    state_path = save_vision_state(
        model,
        output_dir / "vision_state.pt",
        include_projector=args.include_projector,
    )
    config_path = save_vision_config(model, output_dir / "vision_config.json")

    pixel_values = load_static_pixel_values(args.image, STATIC_IMAGE_SIZE)
    torch_dtype = resolve_dtype(args.dtype)
    pixel_values = pixel_values.to(device=args.device, dtype=torch_dtype)

    output, output_kind = run_visual_baseline(
        model,
        pixel_values,
        include_projector=args.include_projector,
    )

    baseline_path = output_dir / "baseline_output.pt"
    torch.save(output.detach().cpu().float(), baseline_path)

    metadata = make_baseline_metadata(
        model_id=args.model_id,
        image_path=args.image,
        input_shape=tuple(pixel_values.shape),
        output_shape=tuple(output.shape),
        output_kind=output_kind,
        dtype=args.dtype,
        device=args.device,
        include_projector=args.include_projector,
    )
    metadata.update(
        {
            "image_size": STATIC_IMAGE_SIZE,
            "vision_state_path": str(state_path),
            "vision_config_path": str(config_path),
            "baseline_output_path": str(baseline_path),
        }
    )

    metadata_path = output_dir / "baseline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
