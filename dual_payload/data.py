"""Folder RGB data and explicit synthetic smoke data; no dataset downloads."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_images(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    paths = sorted({path.resolve() for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES})
    if not paths:
        raise ValueError(f"No supported RGB images found: {root}")
    return paths


def ensure_disjoint(train_paths: list[Path], val_paths: list[Path]) -> None:
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"Train/validation image paths overlap, e.g. {next(iter(overlap))}")


def fixed_message(index: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + index)
    return torch.randint(0, 2, (64,), generator=generator).float()


def _load_rgb_image(path: Path, size: int, crop_generator=None) -> torch.Tensor:
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        scale = size / min(width, height)
        image = image.resize((max(size, round(width * scale)), max(size, round(height * scale))),
                             Image.Resampling.BICUBIC)
        width, height = image.size
        if crop_generator is None:
            left, top = (width - size) // 2, (height - size) // 2
        else:
            left = int(torch.randint(width - size + 1, (1,), generator=crop_generator))
            top = int(torch.randint(height - size + 1, (1,), generator=crop_generator))
        image = image.crop((left, top, left + size, top + size))
        array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_rgb_image(path: str | Path, image_size: int = 256) -> torch.Tensor:
    """Load one explicitly named image using the deterministic validation preprocessing."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Supported RGB image does not exist: {path}")
    if image_size < 8 or image_size % 8:
        raise ValueError("Image size must be a positive multiple of 8")
    return _load_rgb_image(path, image_size)


class ImageFolderDataset(Dataset):
    def __init__(self, directory: str | Path, image_size: int = 256,
                 training: bool = False, seed: int = 2026) -> None:
        self.paths = discover_images(directory)
        self.image_size, self.training, self.seed = image_size, training, seed
        self.epoch = 0
        if image_size < 8 or image_size % 8:
            raise ValueError("Image size must be a positive multiple of 8")

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        path, size = self.paths[index], self.image_size
        generator = (torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index)
                     if self.training else None)
        return {"rgb": _load_rgb_image(path, size, generator),
                "message": fixed_message(index, self.seed), "path": str(path)}


class FixedCartesianDataset(Dataset):
    """Fixed center-cropped image-major Cartesian image/message pairs."""

    def __init__(self, paths: list[Path], messages: torch.Tensor, image_size: int = 256) -> None:
        self.paths = [Path(path).resolve() for path in paths]
        self.messages = messages.detach().cpu().float().clone()
        self.image_size = image_size
        if not self.paths:
            raise ValueError("Fixed Cartesian dataset needs at least one image")
        if (self.messages.ndim != 2 or self.messages.shape[1] != 64
                or not bool(((self.messages == 0) | (self.messages == 1)).all())):
            raise ValueError("Fixed message bank must be a non-empty Nx64 binary tensor")
        if image_size != 256:
            raise ValueError("joint_10x20 fixed center crop requires image_size=256")

    def __len__(self) -> int:
        return len(self.paths) * len(self.messages)

    def __getitem__(self, index: int) -> dict:
        image_index, message_index = divmod(index, len(self.messages))
        path = self.paths[image_index]
        return {"rgb": _load_rgb_image(path, self.image_size),
                "message": self.messages[message_index].clone(), "path": str(path)}


class SyntheticDataset(Dataset):
    """Only for --smoke / tests. Not evidence of natural-image performance."""

    def __init__(self, length: int = 8, image_size: int = 32, seed: int = 2026) -> None:
        self.length, self.image_size, self.seed = length, image_size, seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(self.seed + index)
        texture = torch.rand((3, self.image_size, self.image_size), generator=generator)
        coarse = torch.rand((1, 3, 4, 4), generator=generator)
        smooth = torch.nn.functional.interpolate(coarse, (self.image_size, self.image_size),
                                                  mode="bilinear", align_corners=False)[0]
        return {"rgb": 0.8 * smooth + 0.2 * texture,
                "message": fixed_message(index, self.seed + 100000), "path": f"synthetic:{index}"}
