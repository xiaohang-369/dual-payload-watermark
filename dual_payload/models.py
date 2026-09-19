"""Original four CNNs; defaults are 64 feature channels and eight residual blocks."""

import math

import torch
from torch import Tensor, nn

from .transforms import BlockDCT, rms_cap, ycbcr_to_rgb


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, leaky: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=False) if leaky else nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


def _initialize(module: nn.Module, leaky: bool = False) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if leaky:
                # Explicitly required by the original Watermark Decoder design.
                nn.init.kaiming_normal_(layer.weight, a=0.1, nonlinearity="leaky_relu")
            else:
                # Match the standard Conv2d/Linear fan-in scale for the other trunks.
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _zero_head(head: nn.Conv2d) -> None:
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)


class ColorEncoder(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 8,
                 delta_c: float = 2 / 255, eps: float = 1e-8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.delta_c, self.eps = delta_c, eps
        self.stem = nn.Conv2d(3, channels, 3, padding=1)  # no stem activation
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        _initialize(self)
        _zero_head(self.head)

    def forward(self, y: Tensor, cb: Tensor, cr: Tensor) -> dict[str, Tensor]:
        candidate = self.head(self.body(self.stem(torch.cat((y, cb, cr), dim=1))))
        residual = rms_cap(self.dct.project(candidate, "c"), self.delta_c, self.eps)
        return {"candidate": candidate, "residual": residual, "carrier": y + residual}


class WatermarkEncoder(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 8,
                 delta_w: float = 2 / 255, eps: float = 1e-8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.delta_w, self.eps = delta_w, eps
        self.message_branch = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=False), nn.Linear(128, 64))
        self.image_stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=False))
        self.fusion = nn.Sequential(nn.Conv2d(channels + 64, channels, 3, padding=1), nn.ReLU(inplace=False))
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        _initialize(self)
        _zero_head(self.head)

    def forward(self, s: Tensor, message: Tensor) -> dict[str, Tensor]:
        if message.shape != (s.shape[0], 64):
            raise ValueError("Message must have shape B x 64")
        if not bool(((message == 0) | (message == 1)).all()):
            raise ValueError("Messages must contain only 0 and 1")
        bw = s - self.dct.project(s, "c")
        embedding = self.message_branch(2 * message.to(dtype=s.dtype) - 1)
        message_map = embedding[:, :, None, None].expand(-1, -1, *s.shape[-2:])
        features = self.fusion(torch.cat((self.image_stem(bw), message_map), dim=1))
        candidate = self.head(self.body(features))
        residual = rms_cap(self.dct.project(candidate, "w"), self.delta_w, self.eps)
        return {"candidate": candidate, "residual": residual, "carrier": s + residual}


class ColorDecoder(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.stem = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1), nn.ReLU(inplace=False))
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.chroma_head = nn.Conv2d(channels, 2, 3, padding=1)
        self.luma_head = nn.Conv2d(channels, 1, 3, padding=1)
        _initialize(self)
        nn.init.xavier_normal_(self.chroma_head.weight)
        nn.init.xavier_normal_(self.luma_head.weight)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        z, zc = x - self.dct.project(x, "w"), self.dct.project(x, "c")
        return self.forward_from_components(z, zc)

    def forward_from_components(self, z: Tensor, zc: Tensor) -> dict[str, Tensor]:
        """Decode supplied inputs; used to compare standard and oracle luma paths."""
        if z.shape != zc.shape or z.ndim != 4 or z.shape[1] != 1:
            raise ValueError("z and zc must both have shape B x 1 x H x W")
        features = self.body(self.stem(torch.cat((z, zc), dim=1)))
        cb, cr = self.chroma_head(features).split(1, dim=1)
        raw = self.luma_head(features)
        correction = self.dct.project(raw, "cw")
        y = z + correction
        return {"rgb": ycbcr_to_rgb(y, cb, cr), "y": y, "cb": cb, "cr": cr,
                "raw_luma_delta": raw, "luma_delta": correction, "z": z, "zc": zc}


class WatermarkDecoder(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.stem = nn.Sequential(nn.Conv2d(9, channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=False))
        self.body = nn.Sequential(*(ResidualBlock(channels, leaky=True) for _ in range(blocks)))
        self.head = nn.Conv2d(channels, 64, 1)
        _initialize(self, leaky=True)
        nn.init.xavier_normal_(self.head.weight)

    def forward(self, x: Tensor) -> Tensor:
        evidence = self.head(self.body(self.stem(self.dct.watermark(x))))
        return evidence.mean(dim=(2, 3))  # raw logits: no sigmoid here
