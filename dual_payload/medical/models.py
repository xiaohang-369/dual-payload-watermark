"""Medical V1 interfaces built from the original Ec and Restormer blocks."""

import hashlib
import io
from pathlib import Path

import torch
from torch import Tensor, nn

from ..models import (ColorEncoder as OriginalColorEncoder, RestormerUNet,
                      _blocks, _conv1, _conv3, _initialize_v2)
from ..transforms import BlockDCT, rms_cap, ycbcr_to_rgb
from .profile import ARCHITECTURE, Profile
from .protocol import COLOR_INDICES, PATIENT_INDICES


def _image(x: Tensor):
    if x.ndim != 4 or x.shape[1:] != (1, 256, 256) or x.dtype != torch.float32:
        raise ValueError("Medical V1 expects FP32 Bx1x256x256")
    if not torch.isfinite(x).all():
        raise ValueError("Image tensor contains nonfinite values")


class ColorEncoder(OriginalColorEncoder):
    def __init__(self, color_residual_rms: float):
        super().__init__(delta_c=color_residual_rms)

    def forward(self, y, cb, cr):
        for value in (y, cb, cr):
            _image(value)
        result = super().forward(y, cb, cr)
        # Public Ew must never consume the old plaintext-residual carrier.
        return {"candidate": result["candidate"], "residual": result["residual"]}


class _EmbedBranch(nn.Module):
    def __init__(self, payload_channels, width, out_channels):
        super().__init__()
        self.payload = nn.Sequential(_conv1(payload_channels, width), nn.GELU(), _conv3(width, width))
        self.fusion = _conv1(width + 64, width)
        self.body = _blocks(width, 4, 4, False)
        self.pre_head = nn.Sequential(_conv3(width, width // 2), nn.GELU())
        self.head = _conv1(width // 2, out_channels)

    def forward(self, host, bits):
        features = self.fusion(torch.cat((host, self.payload(2 * bits - 1)), dim=1))
        return self.head(self.pre_head(self.body(features)))


class WatermarkEncoder(nn.Module):
    def __init__(self, color_ciphertext_rms: float, patient_ciphertext_rms: float):
        super().__init__()
        self.dct = BlockDCT()
        self.color_ciphertext_rms = color_ciphertext_rms
        self.patient_ciphertext_rms = patient_ciphertext_rms
        self.host = nn.Sequential(_conv3(64, 64), nn.GELU(), _conv3(64, 64), nn.GELU())
        self.color = _EmbedBranch(236, 128, 39)
        self.patient = _EmbedBranch(2, 64, 9)
        _initialize_v2(self)
        for branch in (self.color, self.patient):
            nn.init.normal_(branch.head.weight, std=1e-4)

    def forward(self, y: Tensor, color_bits: Tensor, patient_bits: Tensor):
        _image(y)
        for bits, channels in ((color_bits, 236), (patient_bits, 2)):
            if bits.shape != (y.shape[0], channels, 32, 32) or not ((bits == 0) | (bits == 1)).all():
                raise ValueError("Invalid binary payload tensor")
        host = self.host(self.dct(y))
        outputs = {}
        for name, branch, bits, indices, limit in (
            ("color", self.color, color_bits, COLOR_INDICES, self.color_ciphertext_rms),
            ("patient", self.patient, patient_bits, PATIENT_INDICES, self.patient_ciphertext_rms)
        ):
            coefficients = branch(host, bits.to(device=y.device, dtype=y.dtype))
            full = y.new_zeros(y.shape[0], 64, 32, 32)
            full[:, list(indices)] = coefficients
            residual = rms_cap(self.dct.inverse(full), limit)
            outputs[name + "_coefficients"] = coefficients
            outputs[name + "_residual"] = residual
        outputs["carrier"] = y + outputs["color_residual"] + outputs["patient_residual"]
        return outputs


class _ExtractBranch(nn.Module):
    def __init__(self, in_channels, width, out_channels):
        super().__init__()
        self.stem = nn.Sequential(_conv1(in_channels, width), nn.GELU(),
                                  _conv3(width, width), nn.GELU())
        self.body = _blocks(width, 4, 4, False)
        self.pre_head = nn.Sequential(_conv3(width, width), nn.GELU())
        self.head = _conv1(width, out_channels)

    def forward(self, x):
        return self.head(self.pre_head(self.body(self.stem(x))))


class WatermarkDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dct = BlockDCT()
        self.color = _ExtractBranch(39, 128, 236)
        self.patient = _ExtractBranch(9, 64, 2)
        _initialize_v2(self)

    def forward(self, gray):
        _image(gray)
        coefficients = self.dct(gray)
        return {"color_logits": self.color(coefficients[:, list(COLOR_INDICES)]),
                "patient_logits": self.patient(coefficients[:, list(PATIENT_INDICES)])}


class ColorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gray_stem = _conv3(1, 8)
        self.residual_stem = _conv3(1, 16)
        self.stem = _conv3(24, 24)
        self.body = RestormerUNet(layer_norm_bias=True)
        self.chroma_head = _conv3(24, 2)
        self.luma_head = _conv3(24, 1)
        _initialize_v2(self)

    def forward(self, gray, residual):
        _image(gray)
        _image(residual)
        if gray.shape != residual.shape:
            raise ValueError("Gray and decrypted residual shapes must match")
        features = self.body(self.stem(torch.cat((self.gray_stem(gray),
                                                 self.residual_stem(residual)), dim=1)))
        cb, cr = self.chroma_head(features).split(1, dim=1)
        correction = self.luma_head(features)  # Full-band correction in V1.
        y = gray + correction
        return {"rgb": ycbcr_to_rgb(y, cb, cr), "y": y, "cb": cb, "cr": cr,
                "luma_delta": correction}


def build_component(name: str, profile: Profile) -> nn.Module:
    limits = profile.config["rms_limits"]
    factories = {"ec": lambda: ColorEncoder(limits["color_residual"]),
                 "ew": lambda: WatermarkEncoder(limits["color_ciphertext"], limits["patient_ciphertext"]),
                 "dc": ColorDecoder, "dw": WatermarkDecoder}
    return factories[name]()


def load_component(name: str, path: str | Path, profile: Profile, device="cpu") -> nn.Module:
    # Hash and deserialize exactly the same bytes; never load paths supplied by a PNG.
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != profile.config["weights"][name]:
        raise ValueError(f"{name} weight checksum mismatch")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("component") != name:
        raise ValueError("Wrong checkpoint architecture or component")
    model = build_component(name, profile)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def import_v2_color_weights(ec: ColorEncoder, dc: ColorDecoder, v2_state: dict) -> dict:
    """Explicit warm start only. Caller must verify provenance of the V2 checkpoint.

    Ec retains its exact computation. Dc imports only the recovery trunk and
    chroma output head; new input semantics and the full-band luma head start fresh.
    Ew/Dw cannot inherit the original 64-bit transport interface.
    """
    report = {"loaded": [], "initialized": []}
    for name, model, prefix in (("ec", ec, "color_encoder."), ("dc", dc, "color_decoder.")):
        destination = model.state_dict()
        for key, value in destination.items():
            allowed = name == "ec" or key.startswith(("body.", "chroma_head."))
            source = v2_state.get(prefix + key)
            if allowed and source is not None and source.shape == value.shape:
                destination[key] = source
                report["loaded"].append(name + "." + key)
            else:
                report["initialized"].append(name + "." + key)
        model.load_state_dict(destination, strict=True)
    return report
