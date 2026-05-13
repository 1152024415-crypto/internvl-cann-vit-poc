from __future__ import annotations

from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STATIC_IMAGE_SIZE = 448


def load_static_pixel_values(image_path: str | Path, image_size: int = STATIC_IMAGE_SIZE):
    """Load one image as a fixed NCHW tensor: ``1 x 3 x image_size x image_size``."""

    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std

    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor
