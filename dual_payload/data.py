"""Prepared work-image interfaces only. No resize, crop, padding or augmentation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .config import message_bits as validate_message_bits


def fixed_message(index: int, seed: int, message_bits: int = 256):
    validate_message_bits({"message_bits": message_bits})
    generator = torch.Generator().manual_seed(seed + index)
    return torch.randint(0, 2, (256,), generator=generator).float()


def load_rgb_image(path):
    with Image.open(path) as image:
        if image.size != (256, 256) or image.mode != "RGB":
            raise ValueError("Input must already be an RGB 256x256 work image; preprocessing is not defined here")
        array = np.array(image, dtype=np.float32) / 255.
    return torch.from_numpy(array).permute(2, 0, 1)


def ensure_disjoint(train_paths, val_paths):
    if {Path(p).resolve() for p in train_paths} & {Path(p).resolve() for p in val_paths}:
        raise ValueError("Training and validation paths overlap")


class ManifestDataset(Dataset):
    """JSON {samples:[{path, patient_id?, message?}]}; paths relative to manifest.

    Patient grouping/splitting is supplied upstream, never decided in this class.
    Validation messages can be explicit or deterministic from seed and row index.
    """
    def __init__(self, manifest, *, training=False, seed=2026):
        self.manifest = Path(manifest).resolve()
        self.rows = json.loads(self.manifest.read_text(encoding="utf-8"))["samples"]
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError("Manifest must contain a nonempty samples list")
        self.paths = [(self.manifest.parent / row["path"]).resolve() for row in self.rows]
        self.training, self.seed = training, seed
        for row in self.rows:
            if "message" in row:
                value = torch.tensor(row["message"])
                if value.shape != (256,) or not bool(((value == 0) | (value == 1)).all()):
                    raise ValueError("Manifest message must contain 256 binary bits")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.training:
            message = torch.randint(0, 2, (256,)).float()
        elif "message" in self.rows[index]:
            message = torch.tensor(self.rows[index]["message"], dtype=torch.float32)
        else:
            message = fixed_message(index, self.seed)
        return {"rgb": load_rgb_image(self.paths[index]), "message": message,
                "path": str(self.paths[index])}


class SyntheticDataset(Dataset):
    def __init__(self, count=2, *, seed=2026, training=False):
        self.count, self.seed, self.training = count, seed, training

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rgb = torch.rand(3, 256, 256, generator=torch.Generator().manual_seed(self.seed + index))
        message = (torch.randint(0, 2, (256,)).float() if self.training
                   else fixed_message(index, self.seed + 10000))
        return {"rgb": rgb, "message": message, "path": f"synthetic:{index}"}
