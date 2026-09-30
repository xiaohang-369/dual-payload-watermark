"""Real file sender/receiver. Encryption and byte transport are outside autograd."""

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from torch import nn

from ..transforms import rgb_to_ycbcr
from .crypto import NonceStore, decrypt_frame, encrypt_frame
from .ldpc import TransportCodec
from .png import AUTH_ITXT_BYTES, hospital_key_id, save_signed_png, verify_png
from .preprocess import integer_pixels
from .profile import Profile
from .protocol import Branch, dequantize_residual, pack_color, quantize_residual, unpack_color


@dataclass
class BranchResult:
    status: str
    reason: str = ""
    rgb: np.ndarray | None = None  # Float CHW; keep out-of-range samples for metrics.
    token: bytes | None = None


@dataclass
class ReceiveResult:
    authentication: str
    color: BranchResult
    patient: BranchResult
    reason: str = ""
    image_id: str | None = None
    content_rect: tuple[int, int, int, int] | None = None

    def summary(self):
        return {"authentication": self.authentication, "reason": self.reason,
                "image_id": self.image_id, "content_rect": self.content_rect,
                "color": {"status": self.color.status, "reason": self.color.reason},
                "patient": {"status": self.patient.status, "reason": self.patient.reason}}


class Sender:
    def __init__(self, profile: Profile, ec: nn.Module, ew: nn.Module, nonces: NonceStore,
                 device="cpu"):
        self.profile, self.ec, self.ew, self.nonces = profile, ec.eval(), ew.eval(), nonces
        self.device = device
        self.codec = TransportCodec(profile, device=device)

    @torch.no_grad()
    def send(self, rgb: torch.Tensor, token: bytes, color_key: bytes, patient_key: bytes,
             signing_key: Ed25519PrivateKey, output: str | Path, content_rect=(0, 0, 256, 256), observer=None):
        if rgb.shape != (1, 3, 256, 256) or rgb.dtype != torch.float32 or not (
            torch.isfinite(rgb).all() and ((rgb >= 0) & (rgb <= 1)).all()
        ):
            raise ValueError("Sender requires one normalized FP32 256x256 RGB work image")
        if color_key == patient_key:
            raise ValueError("Color and patient keys must be independent")
        if len(token) != 16:
            raise ValueError("PatientToken must be exactly 16 bytes")
        rgb = rgb.to(self.device)
        y, cb, cr = rgb_to_ycbcr(rgb)
        residual = self.ec(y, cb, cr)["residual"]
        q, quantization_stats = quantize_residual(residual, self.profile.steps)
        image_id = secrets.token_bytes(16)
        color_frame = encrypt_frame(pack_color(q, self.profile.config["quantization_id"]),
                                    color_key, Branch.COLOR, self.profile.profile_id, image_id, self.nonces)
        patient_frame = encrypt_frame(token, patient_key, Branch.PATIENT,
                                      self.profile.profile_id, image_id, self.nonces)
        color_bits = self.codec.encode(color_frame, Branch.COLOR)
        patient_bits = self.codec.encode(patient_frame, Branch.PATIENT)
        if observer is not None:
            observer('reference', {'color_frame': color_frame, 'patient_frame': patient_frame,
                                    'color_bits': color_bits.detach(), 'patient_bits': patient_bits.detach()})
        carrier = self.ew(y, color_bits, patient_bits)["carrier"]
        pixels = integer_pixels(carrier)[0, 0]
        save_signed_png(output, pixels, self.profile, image_id, signing_key, content_rect)
        # Exercise actual file serialization and re-read before reporting a sent file.
        public_key = signing_key.public_key()
        checked = verify_png(output, {hospital_key_id(public_key): public_key}, self.profile)
        if checked.status != "AUTHENTIC" or not np.array_equal(checked.pixels, pixels):
            raise RuntimeError("Saved PNG did not pass sender read-back verification")
        return {"image_id": image_id.hex(), "profile_id": self.profile.profile_id,
                "profile_sha256": self.profile.digest.hex(), "png_bytes": Path(output).stat().st_size,
                "authentication_chunk_bytes": AUTH_ITXT_BYTES,
                "quantization": quantization_stats,
                "carrier_clipped_fraction": float(((carrier < 0) | (carrier > 1)).float().mean()),
                "authentication": checked.status}


class Receiver:
    """Dependencies: PNG, public profile/models/trust set, optional branch keys only.

    Model loading is deferred until after authentication. The loader is supplied
    by the caller from trusted local paths and is never derived from PNG text.
    """

    def __init__(self, profile: Profile, trusted_keys: dict[bytes, Ed25519PublicKey],
                 model_loader: Callable[[str], nn.Module], device="cpu"):
        self.profile, self.trusted_keys = profile, trusted_keys
        self.model_loader, self.device = model_loader, device

    @torch.no_grad()
    def receive(self, png: str | Path, color_key: bytes | None = None,
                patient_key: bytes | None = None, observer=None) -> ReceiveResult:
        auth = verify_png(png, self.trusted_keys, self.profile)
        if auth.status != "AUTHENTIC":
            return ReceiveResult(auth.status, BranchResult("AUTH_BLOCKED"),
                                 BranchResult("AUTH_BLOCKED"), reason=auth.reason)
        result = ReceiveResult("AUTHENTIC", BranchResult("KEY_MISSING"), BranchResult("KEY_MISSING"),
                               image_id=auth.metadata.image_id.hex(), content_rect=auth.metadata.content_rect)
        if color_key is None and patient_key is None:
            return result
        gray = torch.from_numpy(auth.pixels.copy()).unsqueeze(0).unsqueeze(0).to(self.device).float() / 255
        dw = self.model_loader("dw").eval()
        logits = dw(gray)
        if observer is not None:
            observer('logits', {name: value.detach() for name, value in logits.items()})
        codec = TransportCodec(self.profile, device=self.device)
        for branch, name, key in ((Branch.COLOR, "color", color_key),
                                  (Branch.PATIENT, "patient", patient_key)):
            if key is None:
                continue
            try:
                information = codec.decode_information(logits[name + "_logits"], branch)
                if observer is not None:
                    observer(name + '_information', information)
                frame = codec.information_frame(information, branch)
                plaintext = decrypt_frame(frame, key, branch, self.profile.profile_id, auth.metadata.image_id)
                if observer is not None:
                    observer(name + '_aead_verified', np.array(True))
                if branch == Branch.COLOR:
                    q = unpack_color(plaintext, self.profile.config["quantization_id"])
                    residual = dequantize_residual(q, self.profile.steps, self.device)
                    dc = self.model_loader("dc").eval()
                    rgb = dc(gray, residual)["rgb"]
                    if rgb.shape != (1, 3, 256, 256) or not torch.isfinite(rgb).all():
                        raise ValueError("Dc produced invalid RGB")
                    result.color = BranchResult("OK", rgb=rgb[0].cpu().numpy())
                else:
                    result.patient = BranchResult("OK", token=plaintext)
            except InvalidTag:
                setattr(result, name, BranchResult("DECRYPT_FAILED", "AEAD authentication failed"))
            except (ValueError, OSError) as exc:
                setattr(result, name, BranchResult("DECODE_FAILED", str(exc)))
        return result
