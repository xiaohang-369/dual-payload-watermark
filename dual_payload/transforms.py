"""Fixed full-range, zero-centred colour transforms and orthonormal block DCT."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


WATERMARK_COORDS = ((0, 3), (1, 2), (2, 1), (3, 0),
                    (0, 4), (1, 3), (2, 2), (3, 1), (4, 0))


def rgb_to_ycbcr(rgb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Operate on gamma-coded sRGB values; do not linearize or add 0.5."""
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
        raise ValueError("RGB must be a floating B x 3 x H x W tensor")
    r, g, b = rgb.split(1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, (b - y) / 1.772, (r - y) / 1.402


def ycbcr_to_rgb(y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
    """Fixed inverse with algebraically exact green coefficients; no clamp."""
    r = y + 1.402 * cr
    b = y + 1.772 * cb
    g = y - (0.114 * 1.772 / 0.587) * cb - (0.299 * 1.402 / 0.587) * cr
    return torch.cat((r, g, b), dim=1)


def rms_cap(value: Tensor, delta: float, eps: float = 1e-8) -> Tensor:
    """A single differentiable scale per image, NOT a pixelwise amplitude cap."""
    if value.ndim != 4 or delta < 0 or eps <= 0:
        raise ValueError("RMS cap needs BCHW input, delta >= 0 and eps > 0")
    value = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    rms = (value.square().mean(dim=(1, 2, 3), keepdim=True) + eps).sqrt()
    gain = (delta / rms).clamp(max=1.0)
    return value * gain


class BlockDCT(nn.Module):
    """Top-left aligned, non-overlapping 8 x 8 DCT, channel index u*8+v."""

    def __init__(self) -> None:
        super().__init__()
        n = torch.arange(8, dtype=torch.float64)
        u = n[:, None]
        basis = torch.cos(math.pi * (2 * n[None, :] + 1) * u / 16)
        basis[0] *= math.sqrt(1 / 8)
        basis[1:] *= math.sqrt(2 / 8)
        kernels = torch.einsum("ui,vj->uvij", basis, basis).reshape(64, 1, 8, 8)
        self.register_buffer("kernels", kernels.float())
        degree = (n[:, None] + n[None, :]).reshape(64)
        for name, mask in {
            "c": (degree >= 5) & (degree <= 10),
            "w": (degree >= 3) & (degree <= 4),
            "0": (degree <= 2) | (degree >= 11),
            "cw": (degree >= 3) & (degree <= 10),
        }.items():
            self.register_buffer("mask_" + name, mask.reshape(1, 64, 1, 1))
        self.register_buffer("watermark_indices", torch.tensor([u * 8 + v for u, v in WATERMARK_COORDS]))

    @staticmethod
    def _validate(x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 1 or not x.is_floating_point():
            raise ValueError("DCT input must be floating B x 1 x H x W")
        if min(x.shape[-2:]) < 8 or any(size % 8 for size in x.shape[-2:]):
            raise ValueError("DCT H and W must be positive multiples of 8")

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return F.conv2d(x, self.kernels.to(dtype=x.dtype), stride=8)

    def inverse(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 4 or coefficients.shape[1] != 64:
            raise ValueError("Inverse DCT expects B x 64 x Hb x Wb")
        return F.conv_transpose2d(coefficients, self.kernels.to(dtype=coefficients.dtype), stride=8)

    def project(self, x: Tensor, band: str) -> Tensor:
        if band not in ("c", "w", "0", "cw"):
            raise ValueError(f"Unknown DCT band: {band}")
        return self.inverse(self(x) * getattr(self, "mask_" + band))

    def watermark(self, x: Tensor) -> Tensor:
        """The fixed 9-kernel stride-8 front end, with input gradients intact."""
        self._validate(x)
        x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        kernels = self.kernels.index_select(0, self.watermark_indices).to(dtype=x.dtype)
        return F.conv2d(x, kernels, stride=8)
