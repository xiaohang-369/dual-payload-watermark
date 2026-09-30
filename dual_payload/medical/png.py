"""Strict gray8 PNG parsing and canonical Ed25519 whole-image authentication."""

import base64
import binascii
import hashlib
import io
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from PIL import Image, PngImagePlugin

from .profile import Profile

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
AUTH_KEY = "medical_auth_v1"
SIGN_DOMAIN = b"medical-share/sign/v1\x00"
# auth version, algorithm, pixel format, reserved, W,H,profile;
# profile/transport/codec SHA-256, image ID, key ID, content x,y,w,h.
AUTH_HEADER = struct.Struct(">BBBBHHH32s32s32s16s16s4H")
ITXT_PREFIX = AUTH_KEY.encode() + b"\x00\x00\x00\x00\x00"
AUTH_ITXT_BYTES = 12 + len(ITXT_PREFIX) + 4 * ((AUTH_HEADER.size + 64 + 2) // 3)
MAX_PNG_BYTES = 1024 * 1024


def hospital_key_id(public_key: Ed25519PublicKey) -> bytes:
    return hashlib.sha256(public_key.public_bytes_raw()).digest()[:16]


@dataclass(frozen=True)
class AuthMetadata:
    profile_id: int
    profile_digest: bytes
    transport_id: bytes
    codec_id: bytes
    image_id: bytes
    key_id: bytes
    content_rect: tuple[int, int, int, int]

    def encode(self) -> bytes:
        for value, size in ((self.profile_digest, 32), (self.transport_id, 32), (self.codec_id, 32),
                            (self.image_id, 16), (self.key_id, 16)):
            if len(value) != size:
                raise ValueError("Invalid authentication field size")
        x, y, w, h = self.content_rect
        if not all(type(v) is int for v in self.content_rect) or not (
            0 <= x < 256 and 0 <= y < 256 and 0 < w <= 256 - x and 0 < h <= 256 - y
        ):
            raise ValueError("Invalid content rectangle")
        return AUTH_HEADER.pack(1, 1, 1, 0, 256, 256, self.profile_id, self.profile_digest,
                                 self.transport_id, self.codec_id, self.image_id, self.key_id,
                                 *self.content_rect)

    @classmethod
    def decode(cls, raw: bytes):
        fields = AUTH_HEADER.unpack(raw)
        if fields[:6] != (1, 1, 1, 0, 256, 256):
            raise ValueError("Unsupported authentication format")
        result = cls(*fields[6:12], tuple(fields[12:]))
        if result.encode() != raw:
            raise ValueError("Noncanonical authentication header")
        return result


@dataclass
class Authentication:
    status: str
    reason: str = ""
    pixels: np.ndarray | None = None
    metadata: AuthMetadata | None = None


def _pixels(pixels):
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != (256, 256):
        raise ValueError("PNG requires uint8 256x256 single-channel pixels")


def save_signed_png(path: str | Path, pixels: np.ndarray, profile: Profile, image_id: bytes,
                    signing_key: Ed25519PrivateKey, content_rect=(0, 0, 256, 256)) -> AuthMetadata:
    _pixels(pixels)
    metadata = AuthMetadata(profile.profile_id, profile.digest, profile.model_id(("ew", "dw")),
                            profile.model_id(("ec", "dc")), image_id,
                            hospital_key_id(signing_key.public_key()), content_rect)
    header = metadata.encode()
    signature = signing_key.sign(SIGN_DOMAIN + header + pixels.tobytes(order="C"))
    info = PngImagePlugin.PngInfo()
    info.add_itxt(AUTH_KEY, base64.b64encode(header + signature).decode("ascii"), zip=False)
    # Exclusive creation prevents silently replacing another shared image.
    with Path(path).open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG", pnginfo=info)
    return metadata


def _parse_png(raw: bytes):
    if len(raw) > MAX_PNG_BYTES or not raw.startswith(PNG_MAGIC):
        raise ValueError("Invalid or oversized PNG")
    pos, chunks, auth = 8, [], None
    ended_idat = False
    while pos < len(raw):
        if len(raw) - pos < 12:
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", raw, pos)[0]
        kind = raw[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(raw):
            raise ValueError("Truncated PNG chunk data")
        data = raw[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", raw, pos + 8 + length)[0]
        if zlib.crc32(kind + data) & 0xffffffff != crc:
            raise ValueError("PNG CRC mismatch")
        if kind not in (b"IHDR", b"IDAT", b"iTXt", b"IEND"):
            # Includes APNG, tRNS, EXIF, ICC, gamma and alternative text fields.
            raise ValueError(f"Unsupported PNG chunk: {kind!r}")
        if not chunks and kind != b"IHDR":
            raise ValueError("IHDR must be first")
        if kind == b"IHDR":
            if chunks or data != struct.pack(">IIBBBBB", 256, 256, 8, 0, 0, 0, 0):
                raise ValueError("Only noninterlaced single-channel gray8 256x256 PNG is supported")
        elif kind == b"IDAT":
            if ended_idat:
                raise ValueError("Noncontiguous IDAT chunks")
        elif kind == b"iTXt":
            if auth is not None or not data.startswith(ITXT_PREFIX):
                raise ValueError("Duplicate or unsupported authentication metadata")
            text = data[len(ITXT_PREFIX):]
            auth = base64.b64decode(text, validate=True)
            if base64.b64encode(auth) != text or len(auth) != AUTH_HEADER.size + 64:
                raise ValueError("Noncanonical authentication Base64 or size")
        elif kind == b"IEND":
            if data or end != len(raw) or b"IDAT" not in chunks:
                raise ValueError("Invalid IEND or trailing PNG data")
        if b"IDAT" in chunks and kind != b"IDAT":
            ended_idat = True
        chunks.append(kind)
        pos = end
    if not chunks or chunks[-1] != b"IEND":
        raise ValueError("PNG missing IEND")
    # Pillow decodes the exact byte snapshot whose chunk structure was checked.
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "L" or image.size != (256, 256) or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Unsupported PNG pixels")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    pixels.setflags(write=False)
    return pixels, auth


def verify_png(path: str | Path, trusted_keys: dict[bytes, Ed25519PublicKey],
               profile: Profile) -> Authentication:
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_PNG_BYTES + 1)
        pixels, auth = _parse_png(raw)
        if auth is None:
            return Authentication("AUTH_MISSING", "No medical_auth_v1 iTXt field")
        metadata = AuthMetadata.decode(auth[:AUTH_HEADER.size])
    except (OSError, ValueError, struct.error, binascii.Error, SyntaxError) as exc:
        return Authentication("AUTH_FORMAT_ERROR", str(exc))
    key = trusted_keys.get(metadata.key_id)
    if key is None:
        return Authentication("TRUST_KEY_MISSING", "Hospital key ID is not trusted")
    if hospital_key_id(key) != metadata.key_id:
        return Authentication("AUTH_FAILED", "Trusted key ID mismatch")
    try:
        key.verify(auth[AUTH_HEADER.size:], SIGN_DOMAIN + auth[:AUTH_HEADER.size] + pixels.tobytes())
    except InvalidSignature:
        return Authentication("AUTH_FAILED", "Ed25519 signature verification failed")
    if (metadata.profile_id != profile.profile_id or metadata.profile_digest != profile.digest or
        metadata.transport_id != profile.model_id(("ew", "dw")) or
        metadata.codec_id != profile.model_id(("ec", "dc"))):
        return Authentication("AUTH_FORMAT_ERROR", "Signed profile or model does not match trusted configuration")
    return Authentication("AUTHENTIC", pixels=pixels, metadata=metadata)
