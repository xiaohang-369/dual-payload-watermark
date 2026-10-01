"""Retain the full oriented RGB image, resize to fit, then edge-pad to 256."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def prepare_work_image(path: str | Path):
    with Image.open(path) as source:
        if (source.mode != "RGB" or getattr(source, "n_frames", 1) != 1 or
                source.info.get("icc_profile") or "transparency" in source.info):
            raise ValueError("Input must be a single RGB8 sRGB image without ICC or transparency")
        orientation = source.getexif().get(274, 1)
        if orientation not in range(1, 9):
            raise ValueError("Unsupported EXIF orientation")
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        longest = max(width, height)
        new_w, new_h = max(1, round(width * 256 / longest)), max(1, round(height * 256 / longest))
        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        x, y = (256 - new_w) // 2, (256 - new_h) // 2
        work = np.pad(np.asarray(image), ((y, 256 - new_h - y), (x, 256 - new_w - x), (0, 0)),
                      mode="edge")
    tensor = torch.from_numpy(work.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    return tensor, (x, y, new_w, new_h)


def integer_pixels(x: torch.Tensor) -> np.ndarray:
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Output pixels must be finite floating values")
    return torch.round(x.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def ste_gray8(x: torch.Tensor) -> torch.Tensor:
    """Training helper: real gray8 forward, straight-through backward."""
    integer_forward = torch.round(x.clamp(0, 1) * 255) / 255
    # Avoid cancellation error: forward values must equal the saved PNG pixels.
    return integer_forward.detach() + (x - x.detach())
