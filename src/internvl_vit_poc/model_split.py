from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

VISION_PREFIX = "vision_model."
PROJECTOR_PREFIX = "mlp1."


def split_state_dict(
    state_dict: Mapping[str, Any],
    *,
    include_projector: bool = False,
) -> dict[str, Any]:
    """Extract InternVL vision weights from a full chat-model state dict.

    By default, keys are rewritten for loading into a standalone
    ``InternVisionModel``. With ``include_projector=True``, keys keep their
    original module prefixes so the result can represent the full visual path:
    ``vision_model + mlp1``.
    """

    split: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(VISION_PREFIX):
            out_key = key if include_projector else key.removeprefix(VISION_PREFIX)
            split[out_key] = value
        elif include_projector and key.startswith(PROJECTOR_PREFIX):
            split[key] = value
    return split


def disable_flash_attention(model: Any) -> None:
    """Force InternVL vision config toward export-friendly eager attention."""

    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is not None:
        if hasattr(vision_config, "use_flash_attn"):
            vision_config.use_flash_attn = False
        if hasattr(vision_config, "use_fa3"):
            vision_config.use_fa3 = False

    vision_model_config = getattr(getattr(model, "vision_model", None), "config", None)
    if vision_model_config is not None:
        if hasattr(vision_model_config, "use_flash_attn"):
            vision_model_config.use_flash_attn = False
        if hasattr(vision_model_config, "use_fa3"):
            vision_model_config.use_fa3 = False


def save_vision_state(
    model: Any,
    output_path: str | Path,
    *,
    include_projector: bool = False,
) -> Path:
    """Save the split vision state to a Torch checkpoint."""

    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split = split_state_dict(model.state_dict(), include_projector=include_projector)
    torch.save(split, output_path)
    return output_path


def save_vision_config(model: Any, output_path: str | Path) -> Path:
    """Save InternVL's nested vision config as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_config is None:
        raise ValueError("model.config.vision_config is missing")
    output_path.write_text(vision_config.to_json_string(), encoding="utf-8")
    return output_path
