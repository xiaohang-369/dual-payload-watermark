"""Network V2: four fixed-interface Restormer-based payload networks."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .transforms import BlockDCT, rms_cap, ycbcr_to_rgb


def _conv3(in_channels: int, out_channels: int, *, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1,
                     dilation=1, groups=groups)


def _conv1(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, dilation=1)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """Apply channel-only LayerNorm after BCHW -> B(HW)C rearrangement."""

    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.body = WithBiasLayerNorm(channels) if bias else BiasFreeLayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        tokens = self.body(tokens)
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class MDTA(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("MDTA channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = _conv1(channels, 3 * channels)
        self.qkv_dwconv = _conv3(3 * channels, 3 * channels, groups=3 * channels)
        self.project_out = _conv1(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_channels = channels // self.heads
        shape = (batch, self.heads, head_channels, height * width)
        q = F.normalize(q.reshape(shape), dim=-1)
        k = F.normalize(k.reshape(shape), dim=-1)
        v = v.reshape(shape)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class GDFN(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.66) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = _conv1(channels, 2 * hidden)
        self.dwconv = _conv3(2 * hidden, 2 * hidden, groups=2 * hidden)
        self.project_out = _conv1(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        left, right = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, *, layer_norm_bias: bool) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels, layer_norm_bias)
        self.attention = MDTA(channels, heads)
        self.norm2 = LayerNorm2d(channels, layer_norm_bias)
        self.feed_forward = GDFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


def _blocks(channels: int, count: int, heads: int, layer_norm_bias: bool) -> nn.Sequential:
    return nn.Sequential(*(
        TransformerBlock(channels, heads, layer_norm_bias=layer_norm_bias)
        for _ in range(count)
    ))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv3(channels, channels // 2), nn.PixelUnshuffle(2))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv3(channels, 2 * channels), nn.PixelShuffle(2))

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class RestormerUNet(nn.Module):
    """The frozen 24/48/96/192-channel Ec/Dc backbone."""

    def __init__(self, *, layer_norm_bias: bool) -> None:
        super().__init__()
        self.encoder1 = _blocks(24, 1, 1, layer_norm_bias)
        self.down1 = Downsample(24)
        self.encoder2 = _blocks(48, 1, 2, layer_norm_bias)
        self.down2 = Downsample(48)
        self.encoder3 = _blocks(96, 2, 4, layer_norm_bias)
        self.down3 = Downsample(96)
        self.latent = _blocks(192, 3, 8, layer_norm_bias)
        self.up3 = Upsample(192)
        self.reduce3 = _conv1(192, 96)
        self.decoder3 = _blocks(96, 2, 4, layer_norm_bias)
        self.up2 = Upsample(96)
        self.reduce2 = _conv1(96, 48)
        self.decoder2 = _blocks(48, 1, 2, layer_norm_bias)
        self.up1 = Upsample(48)
        self.reduce1 = _conv1(48, 24)
        self.decoder1 = _blocks(24, 1, 1, layer_norm_bias)
        self.refinement = _blocks(24, 1, 1, layer_norm_bias)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))
        latent = self.latent(self.down3(e3))
        d3 = self.decoder3(self.reduce3(torch.cat((self.up3(latent), e3), dim=1)))
        d2 = self.decoder2(self.reduce2(torch.cat((self.up2(d3), e2), dim=1)))
        d1 = self.decoder1(self.reduce1(torch.cat((self.up1(d2), e1), dim=1)))
        return self.refinement(d1)


def _initialize_v2(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (BiasFreeLayerNorm, WithBiasLayerNorm)):
            nn.init.ones_(layer.weight)
            if isinstance(layer, WithBiasLayerNorm):
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, MDTA):
            nn.init.ones_(layer.temperature)


def _require_same_spatial(*values: Tensor) -> None:
    if not values or any(value.ndim != 4 for value in values):
        raise ValueError("Image tensors must use BCHW layout")
    reference = (values[0].shape[0], values[0].shape[-2:])
    if any((value.shape[0], value.shape[-2:]) != reference for value in values[1:]):
        raise ValueError("Image tensors must have matching batch and spatial dimensions")
    if any(size % 8 for size in values[0].shape[-2:]):
        raise ValueError("Network V2 image height and width must be divisible by 8")


class ColorEncoder(nn.Module):
    def __init__(self, delta_c: float = 2 / 255, eps: float = 1e-8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.delta_c, self.eps = delta_c, eps
        self.y_stem = _conv3(1, 8)
        self.chroma_stem = _conv3(2, 16)
        self.stem = _conv3(24, 24)
        self.body = RestormerUNet(layer_norm_bias=True)
        self.head = _conv3(24, 1)
        _initialize_v2(self)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, y: Tensor, cb: Tensor, cr: Tensor) -> dict[str, Tensor]:
        _require_same_spatial(y, cb, cr)
        if y.shape[1] != 1 or cb.shape[1] != 1 or cr.shape[1] != 1:
            raise ValueError("Ec expects one-channel Y, Cb and Cr tensors")
        shallow = torch.cat((self.y_stem(y), self.chroma_stem(torch.cat((cb, cr), dim=1))), dim=1)
        candidate = self.head(self.body(self.stem(shallow)))
        residual = rms_cap(self.dct.project(candidate, "c"), self.delta_c, self.eps)
        return {"candidate": candidate, "residual": residual, "carrier": y + residual}


class WatermarkEncoder(nn.Module):
    def __init__(self, delta_w: float = 2 / 255, eps: float = 1e-8) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.delta_w, self.eps = delta_w, eps
        self.message_branch = nn.Sequential(
            nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 1024)
        )
        self.message_stem = nn.Sequential(_conv3(1, 32), nn.GELU(), _conv3(32, 64))
        self.image_stem = nn.Sequential(_conv3(64, 64), nn.GELU(), _conv3(64, 64))
        self.fusion_in = _conv1(128, 64)
        self.fusion = nn.Sequential(
            _conv3(64, 64, groups=64), nn.GELU(), _conv1(64, 64)
        )
        self.body = _blocks(64, 4, 4, False)
        self.pre_head = nn.Sequential(_conv3(64, 32), nn.GELU())
        self.head = _conv1(32, 9)
        _initialize_v2(self)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head.bias)

    def forward(self, s: Tensor, message: Tensor) -> dict[str, Tensor]:
        if s.ndim != 4 or s.shape[1:] != (1, 256, 256):
            raise ValueError("Ew requires S with shape B x 1 x 256 x 256")
        if message.shape != (s.shape[0], 64):
            raise ValueError("Message must have shape B x 64")
        if not bool(((message == 0) | (message == 1)).all()):
            raise ValueError("Messages must contain only 0 and 1")
        signed_message = 2 * message.to(dtype=s.dtype) - 1
        message_pattern = self.message_branch(signed_message).reshape(s.shape[0], 1, 32, 32)
        message_features = self.message_stem(message_pattern)
        host_coefficients = self.dct(s.detach())
        host_features = self.image_stem(host_coefficients)
        fused = self.fusion_in(torch.cat((host_features, message_features), dim=1))
        fused = fused + self.fusion(fused)
        features = fused + self.body(fused)
        coefficients = self.head(self.pre_head(features))
        dct_residual = torch.zeros_like(host_coefficients).index_copy(
            1, self.dct.watermark_indices, coefficients
        )
        candidate = self.dct.inverse(dct_residual)
        residual = rms_cap(candidate, self.delta_w, self.eps)
        return {"candidate": candidate, "coefficients": coefficients,
                "dct_residual": dct_residual, "residual": residual,
                "carrier": s + residual}


class ColorDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.structure_stem = _conv3(1, 8)
        self.chroma_stem = _conv3(1, 16)
        self.stem = _conv3(24, 24)
        self.body = RestormerUNet(layer_norm_bias=True)
        self.chroma_head = _conv3(24, 2)
        self.luma_head = _conv3(24, 1)
        _initialize_v2(self)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Dc expects a B x 1 x H x W tensor")
        coefficients = self.dct(x)
        z0 = self.dct.inverse(coefficients * self.dct.mask_0)
        zc = self.dct.inverse(coefficients * self.dct.mask_c)
        z = z0 + zc
        shallow = torch.cat((self.structure_stem(z0), self.chroma_stem(zc)), dim=1)
        features = self.body(self.stem(shallow))
        cb, cr = self.chroma_head(features).split(1, dim=1)
        raw = self.luma_head(features)
        correction = self.dct.project(raw, "cw")
        y = z + correction
        return {"rgb": ycbcr_to_rgb(y, cb, cr), "y": y, "cb": cb, "cr": cr,
                "raw_luma_delta": raw, "luma_delta": correction, "z": z, "zc": zc}


class WatermarkDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dct = BlockDCT()
        self.stem = nn.Sequential(_conv1(9, 32), nn.GELU(), _conv3(32, 64))
        self.body = _blocks(64, 4, 4, False)
        self.pattern_head = nn.Sequential(_conv3(64, 32), nn.GELU(), _conv3(32, 1))
        self.message_mlp = nn.Sequential(nn.Linear(1024, 256), nn.GELU())
        self.head = nn.Linear(256, 64)
        _initialize_v2(self)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 256, 256):
            raise ValueError("Dw requires X with shape B x 1 x 256 x 256")
        initial = self.stem(self.dct.watermark(x))
        features = initial + self.body(initial)
        pattern = self.pattern_head(features)
        hidden = self.message_mlp(pattern.flatten(start_dim=1))
        return self.head(hidden)
