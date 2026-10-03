"""Protocol v1 exact binary layout. Untrusted bytes are verified before parsing."""
from dataclasses import dataclass
import math
import struct
import numpy as np
import torch
from .config import validate_protocol
from .crypto import require_bytes, signature_digest

HEADER_SIZE, SELECTOR_SIZE, GRAY_SIZE, SIGNATURE_SIZE = 152, 1024, 262144, 64
SELECTOR_OFFSET, GRAY_OFFSET, SIGNATURE_OFFSET, PACKAGE_SIZE = 152, 1176, 263320, 263384
# (offset, struct format, fixed value). No native alignment or implicit padding.
FIXED_FIELDS = {
    "Magic": (0, "8s", b"DPWPKG01"), "ProtocolVersion": (8, "H", 1),
    "HeaderLength": (10, "H", 152), "Width": (28, "H", 256), "Height": (30, "H", 256),
    "PixelType": (32, "B", 1), "ByteOrder": (33, "B", 1), "PixelLayout": (34, "B", 1),
    "BlockSize": (35, "B", 8), "DCTVersion": (36, "H", 1), "CoefficientOrderVersion": (38, "H", 1),
    "SelectorVersion": (72, "B", 1), "CandidateCount": (73, "B", 15),
    "PayloadFormatVersion": (136, "H", 1), "PatientTokenLength": (138, "H", 16),
    "CiphertextLength": (140, "H", 16), "GCMTagLength": (142, "H", 16), "ECCID": (144, "H", 0),
    "MessageBits": (146, "H", 256), "KDFVersion": (148, "H", 1), "PermutationVersion": (150, "H", 1),
}
VARIABLE_FIELDS = {
    "image_id": (12, "16s"), "model_id": (40, "32s"), "beta_c": (74, "d"),
    "beta_m": (82, "d"), "min_moved_c": (90, "B"), "min_moved_m": (91, "B"),
    "nc": (92, "16s"), "nm": (108, "16s"), "ngcm": (124, "12s"),
}


@dataclass(frozen=True)
class Header:
    image_id: bytes
    model_id: bytes
    beta_c: float
    beta_m: float
    min_moved_c: int
    min_moved_m: int
    nc: bytes
    nm: bytes
    ngcm: bytes

    def validate(self):
        for name, size in (("image_id", 16), ("model_id", 32), ("nc", 16), ("nm", 16), ("ngcm", 12)):
            require_bytes(getattr(self, name), size, name)
        validate_protocol({"candidate_count": 15, "beta_c": self.beta_c, "beta_m": self.beta_m,
                           "min_moved_c": self.min_moved_c, "min_moved_m": self.min_moved_m})

    def encode(self):
        self.validate()
        result = bytearray(HEADER_SIZE)
        for offset, fmt, value in FIXED_FIELDS.values():
            struct.pack_into(">" + fmt, result, offset, value)
        for name, (offset, fmt) in VARIABLE_FIELDS.items():
            value = getattr(self, name)
            if name.startswith("beta") and value == 0:
                value = 0.0  # Canonical positive zero on send.
            struct.pack_into(">" + fmt, result, offset, value)
        return bytes(result)

    @classmethod
    def decode(cls, raw):
        require_bytes(raw, HEADER_SIZE, "H0")
        for name, (offset, fmt, value) in FIXED_FIELDS.items():
            if struct.unpack_from(">" + fmt, raw, offset)[0] != value:
                raise ValueError(f"Unsupported Protocol v1 field: {name}")
        fields = {name: struct.unpack_from(">" + fmt, raw, offset)[0]
                  for name, (offset, fmt) in VARIABLE_FIELDS.items()}
        header = cls(**fields)
        header.validate()
        for value in (header.beta_c, header.beta_m):
            if value == 0 and math.copysign(1, value) < 0:
                raise ValueError("Negative-zero beta is not canonical")
        return header


@dataclass(frozen=True)
class RawPackage:
    """Only split bytes; this object conveys NO authentication or semantic validity."""
    h0bytes: bytes
    selectors: bytes
    gbytes: bytes
    signature: bytes

    def encode(self):
        for value, size, name in ((self.h0bytes, HEADER_SIZE, "H0"), (self.selectors, SELECTOR_SIZE, "selector"),
                                  (self.gbytes, GRAY_SIZE, "G"), (self.signature, SIGNATURE_SIZE, "signature")):
            require_bytes(value, size, name)
        return self.h0bytes + self.selectors + self.gbytes + self.signature


@dataclass(frozen=True)
class VerifiedPackage:
    raw: RawPackage
    header: Header
    gray: torch.Tensor


def gray_to_bytes(gray):
    if gray.shape != (1, 1, 256, 256) or gray.dtype != torch.float32 or not bool(torch.isfinite(gray).all()):
        raise ValueError("G must be finite FP32 [1,1,256,256]")
    return gray.detach().cpu().contiguous().numpy().astype(">f4").tobytes(order="C")


def gray_from_bytes(raw):
    require_bytes(raw, GRAY_SIZE, "G")
    array = np.frombuffer(raw, dtype=">f4").astype(np.float32).reshape(1, 1, 256, 256)
    if not np.isfinite(array).all():
        raise ValueError("G contains nonfinite pixels")
    return torch.from_numpy(array)


def encode_package(header, selectors, gray, signing_key):
    h0 = header.encode()
    require_bytes(selectors, SELECTOR_SIZE, "selector")
    gbytes = gray_to_bytes(gray)
    signature = signing_key.sign(signature_digest(h0, selectors, gbytes))
    return RawPackage(h0, selectors, gbytes, signature).encode()


def decode_package(data):
    """Structural split only; recovery must call verify_package."""
    require_bytes(data, PACKAGE_SIZE, "package")
    return RawPackage(data[:152], data[152:1176], data[1176:263320], data[263320:])


def verify_package(data, trusted_public_key, *, expected_model_id):
    require_bytes(expected_model_id, 32, "expected ModelID")
    raw = decode_package(data)
    trusted_public_key.verify(raw.signature, signature_digest(raw.h0bytes, raw.selectors, raw.gbytes))
    header = Header.decode(raw.h0bytes)
    if header.model_id != expected_model_id:
        raise ValueError("Package ModelID does not match the loaded checkpoint")
    return VerifiedPackage(raw, header, gray_from_bytes(raw.gbytes))
