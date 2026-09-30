"""AES-256-GCM with a persistent per-key nonce reservation ledger."""

import hashlib
import secrets
import sqlite3
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .protocol import Branch, FRAME_BYTES, PLAINTEXT_BYTES, aad, check_header, make_header


class NonceStore:
    """Reserve random 96-bit nonces before encryption, atomically across processes.

    All senders using a given key must share this persistent ledger. Never reset
    or roll it back while retaining its keys. Receivers do not need this file.
    Only SHA-256 key fingerprints and nonces are stored, never secret keys.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS nonces (key_hash BLOB, nonce BLOB, "
                       "PRIMARY KEY(key_hash, nonce))")

    def reserve(self, key: bytes) -> bytes:
        _check_key(key)
        fingerprint = hashlib.sha256(key).digest()
        for _ in range(8):
            nonce = secrets.token_bytes(12)
            try:
                with sqlite3.connect(self.path, timeout=30) as db:
                    db.execute("INSERT INTO nonces VALUES (?, ?)", (fingerprint, nonce))
                return nonce
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Could not reserve a unique nonce")


def _check_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")


def encrypt_frame(plaintext: bytes, key: bytes, branch: Branch, profile_id: int,
                  image_id: bytes, nonces: NonceStore) -> bytes:
    _check_key(key)
    if len(plaintext) != PLAINTEXT_BYTES[branch]:
        raise ValueError("Wrong branch plaintext length")
    header = make_header(branch, profile_id, image_id)
    nonce = nonces.reserve(key)
    return header + nonce + AESGCM(key).encrypt(nonce, plaintext, aad(header, branch))


def decrypt_frame(frame: bytes, key: bytes, branch: Branch, profile_id: int,
                  image_id: bytes) -> bytes:
    _check_key(key)
    if len(frame) != FRAME_BYTES[branch]:
        raise ValueError("Wrong branch frame length")
    check_header(frame[:32], branch, profile_id, image_id)
    # AESGCM releases plaintext only after authenticating the entire message.
    return AESGCM(key).decrypt(frame[32:44], frame[44:], aad(frame[:32], branch))
