from copy import deepcopy

import numpy as np
import pytest
import torch
from cryptography.exceptions import InvalidTag

from dual_payload.medical.crypto import NonceStore, decrypt_frame, encrypt_frame
from dual_payload.medical.ldpc import TransportCodec
from dual_payload.medical.profile import load_profile, register_profile, template, validate_config
from dual_payload.medical.protocol import (BLOCKS, CHANNELS, COLOR_INDICES, FRAME_BYTES,
                                           PLAINTEXT_BYTES, Branch, bits_to_bytes, bytes_to_bits,
                                           dequantize_residual, pack_color, quantize_residual,
                                           unpack_color)
from dual_payload.transforms import BlockDCT


def test_msb_first_twos_complement_and_exact_packet():
    q = np.resize(np.arange(-8, 8, dtype=np.int8), (32, 32, 39))
    packet = pack_color(q, 7)
    assert len(packet) == 19992
    assert packet[:24].hex() == "4d434c52000100070100010000009c000002700000000000"
    assert packet[24:32].hex() == "89abcdef01234567"
    assert np.array_equal(unpack_color(packet, 7), q)
    assert bytes_to_bits(b"\x81").tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    assert bits_to_bytes(bytes_to_bits(packet)) == packet
    for broken in (packet[:-1], packet + b"\0", packet[:23] + b"\1" + packet[24:]):
        with pytest.raises(ValueError):
            unpack_color(broken, 7)
    with pytest.raises(ValueError):
        unpack_color(packet, 8)


def test_dct_frequency_order_and_residual_roundtrip(profile):
    q = np.random.default_rng(2).integers(-8, 8, size=(32, 32, 39), dtype=np.int8)
    residual = dequantize_residual(q, profile.steps)
    coefficients = BlockDCT()(residual)
    expected = torch.tensor(q).permute(2, 0, 1) * torch.tensor(profile.steps)[:, None, None]
    torch.testing.assert_close(coefficients[0, list(COLOR_INDICES)], expected, atol=3e-7, rtol=1e-5)
    recovered, stats = quantize_residual(residual, profile.steps)
    assert np.array_equal(recovered, q)
    assert stats["clipped_fraction"] == 0
    assert residual.min() < 0  # No [0,1] clamp or +0.5 offset.


@pytest.mark.parametrize("branch", list(Branch))
def test_real_ldpc_frame_layout_roundtrip_and_soft_input(profile, branch):
    codec = TransportCodec(profile)
    frame = np.random.default_rng(4).bytes(FRAME_BYTES[branch])
    layout = codec.encode(frame, branch)
    assert layout.shape == (1, CHANNELS[branch], 32, 32)
    assert layout.numel() - BLOCKS[branch] * 1536 == 512
    assert torch.count_nonzero(layout.flatten()[-512:]) == 0
    logits = (layout * 2 - 1) * 12
    logits.flatten()[-512:] = 1234  # Padding never contributes to decoding.
    assert codec.decode(logits, branch) == frame
    # Recover isolated uncertain/wrong-sign bits with the actual decoder.
    logits.flatten()[::3000] *= -0.02
    assert codec.decode(logits, branch) == frame
    permutation = profile.permutation(branch)
    inverse = profile.permutation(branch, inverse=True)
    assert np.array_equal(permutation[inverse], np.arange(permutation.size))
    assert min(np.unique(inverse[i:i + 1536] % 1024).size
               for i in range(0, len(inverse), 1536)) > 1


@pytest.mark.parametrize("branch", list(Branch))
def test_encryption_binds_every_header_field_and_ciphertext(tmp_path, branch):
    store = NonceStore(tmp_path / "nonces.sqlite")
    key, image_id = b"k" * 32, b"i" * 16
    plain = bytes(PLAINTEXT_BYTES[branch])
    frame = encrypt_frame(plain, key, branch, 3, image_id, store)
    assert len(frame) == FRAME_BYTES[branch]
    assert decrypt_frame(frame, key, branch, 3, image_id) == plain
    for index in (0, 1, 2, 4, 20, 24, 32, 44, len(frame) - 1):
        changed = bytearray(frame)
        changed[index] ^= 1
        with pytest.raises((ValueError, InvalidTag)):
            decrypt_frame(bytes(changed), key, branch, 3, image_id)
    with pytest.raises(InvalidTag):
        decrypt_frame(frame, b"w" * 32, branch, 3, image_id)


def test_nonce_ledger_survives_reopen_and_rejects_collisions(tmp_path, monkeypatch):
    path = tmp_path / "nonces.sqlite"
    monkeypatch.setattr("dual_payload.medical.crypto.secrets.token_bytes", lambda n: bytes(n))
    assert NonceStore(path).reserve(b"k" * 32) == bytes(12)
    with pytest.raises(RuntimeError, match="unique nonce"):
        NonceStore(path).reserve(b"k" * 32)
    assert NonceStore(path).reserve(b"j" * 32) == bytes(12)


def test_profile_fails_closed_and_id_is_immutable(test_config, tmp_path):
    with pytest.raises(ValueError):
        validate_config(template())
    with pytest.raises(ValueError, match="Test profile"):
        validate_config(test_config)
    path = register_profile(test_config, tmp_path, allow_test=True)
    original = load_profile(path, allow_test=True)
    assert register_profile(test_config, tmp_path, allow_test=True) == path
    changed = deepcopy(test_config)
    changed["quantization_steps"][0] *= 2
    with pytest.raises(ValueError, match="already registered"):
        register_profile(changed, tmp_path, allow_test=True)
    assert load_profile(path, allow_test=True).digest == original.digest
    array_file = path / "color-permutation.u32be"
    data = bytearray(array_file.read_bytes())
    data[0] ^= 1
    array_file.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        load_profile(path, allow_test=True)
