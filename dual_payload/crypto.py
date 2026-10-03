"""Frozen Protocol v1 key derivation and patient payload primitives."""
import hashlib
import hmac
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

COLOR_INFO = b"DPW-COLOR-PERM-v1"
PATIENT_INFO = b"DPW-PATIENT-PERM-v1"
ENCRYPTION_INFO = b"DPW-PATIENT-ENC-v1"


def require_bytes(value, length, name):
    if type(value) is not bytes or len(value) != length:
        raise ValueError(f"{name} must be {length} raw bytes")
    return value


def derive_key(master: bytes, salt: bytes, info: bytes) -> bytes:
    require_bytes(master, 32, "master key")
    require_bytes(salt, 16, "HKDF salt")
    if info not in (COLOR_INFO, PATIENT_INFO, ENCRYPTION_INFO):
        raise ValueError("Unknown Protocol v1 HKDF domain")
    prk = hmac.digest(salt, master, "sha256")
    return hmac.digest(prk, info + b"\x01", "sha256")


def encrypt_patient(token, km, nm, nonce, h0bytes):
    require_bytes(token, 16, "PatientToken")
    require_bytes(nonce, 12, "NGCM")
    require_bytes(h0bytes, 152, "H0")
    return AESGCM(derive_key(km, nm, ENCRYPTION_INFO)).encrypt(nonce, token, h0bytes)


def decrypt_patient(payload, km, nm, nonce, h0bytes):
    require_bytes(payload, 32, "CT || Tag")
    require_bytes(nonce, 12, "NGCM")
    require_bytes(h0bytes, 152, "H0")
    # InvalidTag propagates. No token is released before successful authentication.
    return AESGCM(derive_key(km, nm, ENCRYPTION_INFO)).decrypt(nonce, payload, h0bytes)


def payload_to_bits(payload):
    require_bytes(payload, 32, "CT || Tag")
    return torch.tensor([(byte >> (7 - bit)) & 1 for byte in payload for bit in range(8)],
                        dtype=torch.float32)


def logits_to_payload(logits):
    if logits.shape != (256,) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Dw must return exactly 256 finite logits")
    bits = (logits.detach().cpu() >= 0).tolist()
    return bytes(sum(int(bits[8 * j + k]) << (7 - k) for k in range(8)) for j in range(32))


def signature_digest(h0bytes, selectors, gbytes):
    return hashlib.sha256(b"DPWSIG01" + h0bytes + selectors + gbytes).digest()
