"""Fixed wire constants, MSB-first bytes/bits and signed 4-bit color packets."""

import struct
from enum import IntEnum

import numpy as np
import torch

from ..transforms import BlockDCT, WATERMARK_COORDS


class Branch(IntEnum):
    COLOR = 1
    PATIENT = 2


VERSION = 1
COLOR_MAGIC = b"MCLR"
COLOR_INDICES = tuple(u * 8 + v for u in range(8) for v in range(8) if 5 <= u + v <= 10)
PATIENT_INDICES = tuple(u * 8 + v for u, v in WATERMARK_COORDS)
COEFFICIENTS = 32 * 32 * 39
COLOR_HEADER = struct.Struct(">4sHHHHIII")
FRAME_HEADER = struct.Struct(">BBH16sI8s")
PLAINTEXT_BYTES = {Branch.COLOR: 19992, Branch.PATIENT: 16}
FRAME_BYTES = {b: n + 60 for b, n in PLAINTEXT_BYTES.items()}
BLOCKS = {Branch.COLOR: 157, Branch.PATIENT: 1}
CHANNELS = {Branch.COLOR: 236, Branch.PATIENT: 2}
AAD_DOMAIN = b"medical-share/aead/v1\x00"


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size % 8 or not np.isin(bits, (0, 1)).all():
        raise ValueError("Expected a byte-aligned vector of binary bits")
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def pack_color(q: np.ndarray, quantization_id: int) -> bytes:
    """Input layout: block-row, block-column, ascending frequency (32,32,39)."""
    q = np.asarray(q)
    if q.shape != (32, 32, 39) or not np.issubdtype(q.dtype, np.integer):
        raise ValueError("Color integers must have shape (32,32,39)")
    if q.min() < -8 or q.max() > 7:
        raise ValueError("4-bit coefficients must be in [-8,7]")
    nibbles = (q.reshape(-1).astype(np.int16) & 15).astype(np.uint8)
    header = COLOR_HEADER.pack(COLOR_MAGIC, VERSION, quantization_id, 256, 256,
                               COEFFICIENTS, COEFFICIENTS * 4, 0)
    return header + ((nibbles[::2] << 4) | nibbles[1::2]).tobytes()


def unpack_color(packet: bytes, quantization_id: int) -> np.ndarray:
    if len(packet) != PLAINTEXT_BYTES[Branch.COLOR]:
        raise ValueError("Wrong color packet length")
    if COLOR_HEADER.unpack_from(packet) != (
        COLOR_MAGIC, VERSION, quantization_id, 256, 256, COEFFICIENTS, COEFFICIENTS * 4, 0
    ):
        raise ValueError("Invalid color header")
    data = np.frombuffer(packet[24:], dtype=np.uint8)
    q = np.stack((data >> 4, data & 15), axis=1).reshape(-1).astype(np.int8)
    q[q >= 8] -= 16
    return q.reshape(32, 32, 39)


def quantize_residual(residual: torch.Tensor, steps: tuple[float, ...]):
    if residual.shape != (1, 1, 256, 256) or not torch.isfinite(residual).all():
        raise ValueError("Residual must be finite 1x1x256x256")
    dct = BlockDCT().to(residual.device)
    coefficients = dct(residual)[0, list(COLOR_INDICES)].permute(1, 2, 0)
    delta = torch.as_tensor(steps, dtype=coefficients.dtype, device=residual.device)
    rounded = torch.round(coefficients / delta)  # ties to even
    q = rounded.clamp(-8, 7)
    stats = {"clipped_fraction": float(((rounded < -8) | (rounded > 7)).float().mean()),
             "coefficient_mse": float((coefficients - q * delta).square().mean())}
    return q.detach().cpu().numpy().astype(np.int8), stats


def dequantize_residual(q: np.ndarray, steps: tuple[float, ...], device="cpu") -> torch.Tensor:
    # No receiver-side rescaling, offset or clamp.
    if np.asarray(q).shape != (32, 32, 39):
        raise ValueError("Wrong coefficient shape")
    selected = torch.as_tensor(q, dtype=torch.float32, device=device).permute(2, 0, 1)
    selected = selected * torch.tensor(steps, device=device)[:, None, None]
    full = torch.zeros(1, 64, 32, 32, device=device)
    full[:, list(COLOR_INDICES)] = selected
    return BlockDCT().to(device).inverse(full)


def make_header(branch: Branch, profile_id: int, image_id: bytes) -> bytes:
    if len(image_id) != 16:
        raise ValueError("image_id must be 16 bytes")
    return FRAME_HEADER.pack(VERSION, branch, profile_id, image_id,
                             PLAINTEXT_BYTES[branch] + 28, bytes(8))


def check_header(header: bytes, branch: Branch, profile_id: int, image_id: bytes) -> None:
    if header != make_header(branch, profile_id, image_id):
        raise ValueError("Frame version, branch, profile, image ID, length or reserved bytes mismatch")


def aad(header: bytes, branch: Branch) -> bytes:
    return AAD_DOMAIN + (b"COLOR\x00" if branch == Branch.COLOR else b"PATIENT\x00") + header
