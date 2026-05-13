from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _shape_to_list(shape: Iterable[int]) -> list[int]:
    return [int(dim) for dim in shape]


def make_baseline_metadata(
    *,
    model_id: str,
    image_path: str | Path,
    input_shape: Iterable[int],
    output_shape: Iterable[int],
    output_kind: str,
    dtype: str,
    device: str,
    include_projector: bool,
) -> dict[str, object]:
    """Build stable metadata for a saved PyTorch baseline tensor."""

    return {
        "model_id": model_id,
        "image_path": str(image_path),
        "input_shape": _shape_to_list(input_shape),
        "output_shape": _shape_to_list(output_shape),
        "output_kind": output_kind,
        "dtype": dtype,
        "device": device,
        "include_projector": bool(include_projector),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
