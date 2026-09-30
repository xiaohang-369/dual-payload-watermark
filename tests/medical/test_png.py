import base64
import struct
import zlib

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image, PngImagePlugin

from dual_payload.medical.png import (AUTH_HEADER, AUTH_KEY, ITXT_PREFIX, hospital_key_id,
                                      save_signed_png, verify_png)


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


@pytest.fixture
def signed(profile, tmp_path):
    key = Ed25519PrivateKey.generate()
    pixels = np.random.default_rng(17).integers(0, 256, (256, 256), dtype=np.uint8)
    path = tmp_path / "shared.png"
    save_signed_png(path, pixels, profile, b"i" * 16, key, (0, 64, 256, 128))
    trusted = {hospital_key_id(key.public_key()): key.public_key()}
    return path, pixels, trusted


def test_saved_pixels_identical_and_recompression_authenticates(signed, profile, tmp_path):
    path, pixels, trusted = signed
    result = verify_png(path, trusted, profile)
    assert result.status == "AUTHENTIC"
    assert np.array_equal(result.pixels, pixels)
    assert result.metadata.content_rect == (0, 64, 256, 128)
    with Image.open(path) as image:
        info = PngImagePlugin.PngInfo()
        info.add_itxt(AUTH_KEY, image.info[AUTH_KEY])
    new = tmp_path / "recompressed.png"
    Image.fromarray(pixels).save(new, compress_level=0, pnginfo=info)
    assert verify_png(new, trusted, profile).status == "AUTHENTIC"
    pixels[12, 13] ^= 1
    Image.fromarray(pixels).save(new, pnginfo=info)
    assert verify_png(new, trusted, profile).status == "AUTH_FAILED"


def test_missing_auth_and_untrusted_key(signed, profile, tmp_path):
    path, pixels, trusted = signed
    assert verify_png(path, {}, profile).status == "TRUST_KEY_MISSING"
    missing = tmp_path / "no-auth.png"
    Image.fromarray(pixels).save(missing)
    assert verify_png(missing, trusted, profile).status == "AUTH_MISSING"


@pytest.mark.parametrize("kind,data", [
    (b"tRNS", b"\x00\x00"), (b"acTL", struct.pack(">II", 1, 0)),
    (b"eXIf", b"fake"), (b"gAMA", struct.pack(">I", 45455)),
    (b"sRGB", b"\0"), (b"iCCP", b"fake"), (b"tEXt", b"medical_auth_v1\0fake")
])
def test_unagreed_display_and_animation_chunks_rejected(signed, profile, tmp_path, kind, data):
    path, _, trusted = signed
    raw = path.read_bytes()
    changed = tmp_path / "extra.png"
    changed.write_bytes(raw[:33] + chunk(kind, data) + raw[33:])
    assert verify_png(changed, trusted, profile).status == "AUTH_FORMAT_ERROR"


def test_duplicate_itxt_bad_crc_and_trailing_bytes_rejected(signed, profile, tmp_path):
    path, _, trusted = signed
    raw = path.read_bytes()
    with Image.open(path) as image:
        auth = image.info[AUTH_KEY].encode()
    bad_crc = bytearray(raw)
    bad_crc[-1] ^= 1
    for index, invalid in enumerate((raw[:33] + chunk(b"iTXt", ITXT_PREFIX + auth) + raw[33:],
                                     raw + b"trailing", bytes(bad_crc), raw[:-1])):
        changed = tmp_path / f"invalid-{index}.png"
        changed.write_bytes(invalid)
        assert verify_png(changed, trusted, profile).status == "AUTH_FORMAT_ERROR"


@pytest.mark.parametrize("mode,shape,dtype", [("RGB", (256, 256, 3), np.uint8),
                                              ("LA", (256, 256, 2), np.uint8),
                                              ("I;16", (256, 256), np.uint16),
                                              ("L", (128, 256), np.uint8)])
def test_wrong_pixel_formats_rejected(profile, tmp_path, mode, shape, dtype):
    path = tmp_path / "wrong.png"
    Image.fromarray(np.zeros(shape, dtype=dtype)).save(path)
    assert verify_png(path, {}, profile).status == "AUTH_FORMAT_ERROR"


@pytest.mark.parametrize("offset", [6, 12, 50, 115, AUTH_HEADER.size - 1, AUTH_HEADER.size])
def test_modified_metadata_is_not_accepted(signed, profile, tmp_path, offset):
    path, pixels, trusted = signed
    with Image.open(path) as image:
        auth = bytearray(base64.b64decode(image.info[AUTH_KEY]))
    auth[offset] ^= 1
    info = PngImagePlugin.PngInfo()
    info.add_itxt(AUTH_KEY, base64.b64encode(auth).decode())
    changed = tmp_path / "changed-metadata.png"
    Image.fromarray(pixels).save(changed, pnginfo=info)
    assert verify_png(changed, trusted, profile).status != "AUTHENTIC"
