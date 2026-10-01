import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image
from torch import nn

from dual_payload.medical.crypto import NonceStore, encrypt_frame
from dual_payload.medical.ldpc import TransportCodec
from dual_payload.medical.models import (ColorDecoder, ColorEncoder, WatermarkDecoder,
                                         WatermarkEncoder, build_component,
                                         load_component)
from dual_payload.medical.pipeline import Receiver, Sender
from dual_payload.medical.png import hospital_key_id, save_signed_png
from dual_payload.medical.preprocess import integer_pixels, prepare_work_image, ste_gray8
from dual_payload.medical.profile import ARCHITECTURE, load_profile, register_profile
from dual_payload.medical.protocol import Branch, dequantize_residual, pack_color
from dual_payload.transforms import BlockDCT, rgb_to_ycbcr


def test_four_network_shapes_bands_dual_input_and_gradients(profile):
    ec, ew, dc, dw = (build_component(name, profile) for name in ("ec", "ew", "dc", "dw"))
    rgb = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        color = ec(*rgb_to_ycbcr(rgb))
        assert "carrier" not in color
    gray = torch.rand(1, 1, 256, 256)
    cbits = torch.randint(0, 2, (1, 236, 32, 32)).float()
    mbits = torch.randint(0, 2, (1, 2, 32, 32)).float()
    embedded = ew(gray, cbits, mbits)
    dct = BlockDCT()
    for name, band in (("color", "c"), ("patient", "w")):
        residual = embedded[name + "_residual"]
        assert float(residual.detach().square().mean().sqrt()) <= 2 / 255 + 1e-7
        torch.testing.assert_close(dct.project(residual, band), residual, atol=1e-8, rtol=1e-4)
    received = ste_gray8(embedded["carrier"])
    logits = dw(received)
    assert logits["color_logits"].shape == cbits.shape
    assert logits["patient_logits"].shape == mbits.shape
    loss = sum(nn.functional.binary_cross_entropy_with_logits(logits[n + "_logits"], b)
               for n, b in (("color", cbits), ("patient", mbits)))
    loss.backward()
    for head in (ew.color.head, ew.patient.head, dw.color.head, dw.patient.head):
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    # Dc receives actual quantized gray and independent decrypted residual.
    residual = torch.randn_like(gray, requires_grad=True)
    output = dc(received.detach(), residual)
    assert output["rgb"].shape == rgb.shape
    output["rgb"].mean().backward()
    assert residual.grad.abs().sum() > 0
    # Raw logits must remain negative when the head is explicitly biased negative.
    with torch.no_grad():
        dw.patient.head.weight.zero_()
        dw.patient.head.bias.fill_(-3)
        assert torch.equal(dw(gray)["patient_logits"], torch.full_like(mbits, -3))


def test_scratch_initialization_has_nonzero_residual_and_zero_luma_correction(profile):
    ec, dc = build_component("ec", profile), build_component("dc", profile)
    with torch.no_grad():
        gray = torch.rand(1, 1, 256, 256)
        residual = ec(*rgb_to_ycbcr(torch.rand(1, 3, 256, 256)))['residual']
        assert torch.isfinite(residual).all() and residual.abs().sum() > 0
        recovered = dc(gray, residual)
        assert torch.count_nonzero(recovered['luma_delta']) == 0
        assert torch.equal(recovered['y'], gray)


def test_preprocess_preserves_whole_image_and_integer_rounding(tmp_path):
    pixels = np.zeros((50, 100, 3), dtype=np.uint8)
    pixels[:, :50, 0] = 255
    pixels[:, 50:, 2] = 255
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    rgb, rect = prepare_work_image(path)
    assert rgb.shape == (1, 3, 256, 256) and rect == (0, 64, 256, 128)
    assert torch.equal(rgb[:, :, 0], rgb[:, :, 64])
    assert rgb[0, 0, 128, 0] == 1 and rgb[0, 2, 128, -1] == 1
    x = torch.tensor([-1, 0.5 / 255, 1.5 / 255, 2.5 / 255, 2.0])
    assert integer_pixels(x).tolist() == [0, 0, 2, 2, 255]
    assert np.array_equal(integer_pixels(ste_gray8(x)), integer_pixels(x))


class IdealLogits(nn.Module):
    """Test-only injection: verify protocol/permissions, not neural extraction."""

    def __init__(self, color, patient):
        super().__init__()
        self.values = {"color_logits": 16 * (2 * color - 1),
                       "patient_logits": 16 * (2 * patient - 1)}
        self.seen = None

    def forward(self, gray):
        self.seen = gray.clone()
        return self.values


class ResidualProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, gray, residual):
        self.inputs = (gray.clone(), residual.clone())
        return {"rgb": gray.expand(-1, 3, -1, -1) + residual}


@pytest.fixture(scope="module")
def ideal_channel(profile, tmp_path_factory):
    directory = tmp_path_factory.mktemp("ideal-channel")
    color_key, patient_key, token, image_id = b"c" * 32, b"m" * 32, b"t" * 16, b"i" * 16
    q = np.random.default_rng(4).integers(-8, 8, (32, 32, 39), dtype=np.int8)
    store, codec = NonceStore(directory / "nonce.sqlite"), TransportCodec(profile)
    frames = [encrypt_frame(plain, key, branch, profile.profile_id, image_id, store)
              for plain, key, branch in ((pack_color(q, profile.config["quantization_id"]), color_key,
                                          Branch.COLOR), (token, patient_key, Branch.PATIENT))]
    extractor = IdealLogits(codec.encode(frames[0], Branch.COLOR), codec.encode(frames[1], Branch.PATIENT))
    pixels = np.random.default_rng(5).integers(0, 256, (256, 256), dtype=np.uint8)
    hospital = Ed25519PrivateKey.generate()
    path = directory / "shared.png"
    save_signed_png(path, pixels, profile, image_id, hospital)
    return dict(path=path, pixels=pixels, q=q, extractor=extractor,
                trusted={hospital_key_id(hospital.public_key()): hospital.public_key()},
                color_key=color_key, patient_key=patient_key, token=token)


@pytest.mark.parametrize("has_color,has_patient", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_permissions_with_real_encrypted_frames(profile, ideal_channel, has_color, has_patient):
    data = ideal_channel
    probe, requested = ResidualProbe(), []

    def loader(name):
        requested.append(name)
        return data["extractor"] if name == "dw" else probe

    result = Receiver(profile, data["trusted"], loader).receive(
        data["path"], data["color_key"] if has_color else None, data["patient_key"] if has_patient else None)
    assert result.authentication == "AUTHENTIC"
    assert result.color.status == ("OK" if has_color else "KEY_MISSING")
    assert result.patient.status == ("OK" if has_patient else "KEY_MISSING")
    assert result.patient.token == (data["token"] if has_patient else None)
    assert (result.color.rgb is not None) == has_color
    if has_color:
        torch.testing.assert_close(probe.inputs[1], dequantize_residual(data["q"], profile.steps))
        torch.testing.assert_close(probe.inputs[0][0, 0], torch.tensor(data["pixels"]).float() / 255)
    if not has_color and not has_patient:
        assert requested == []


@pytest.mark.parametrize("wrong_branch", ["color", "patient"])
def test_wrong_key_does_not_release_partial_plaintext_or_erase_other_branch(profile, ideal_channel, wrong_branch):
    data = ideal_channel
    result = Receiver(profile, data["trusted"],
                      lambda name: data["extractor"] if name == "dw" else ResidualProbe()).receive(
        data["path"], b"x" * 32 if wrong_branch == "color" else data["color_key"],
        b"x" * 32 if wrong_branch == "patient" else data["patient_key"])
    failed = getattr(result, wrong_branch)
    assert failed.status == "DECRYPT_FAILED" and failed.rgb is None and failed.token is None
    assert getattr(result, "patient" if wrong_branch == "color" else "color").status == "OK"


def test_auth_failure_never_enters_network(profile, ideal_channel, tmp_path):
    def forbidden(name):
        raise AssertionError("Authentication must precede model loading")
    path = tmp_path / "plain.png"
    Image.fromarray(ideal_channel["pixels"]).save(path)
    result = Receiver(profile, ideal_channel["trusted"], forbidden).receive(path, b"c" * 32, b"m" * 32)
    assert result.authentication == "AUTH_MISSING"
    assert result.color.status == result.patient.status == "AUTH_BLOCKED"


def test_real_sender_and_separate_receiver_process_have_no_private_sender_dependency(
    profile, test_config, tmp_path
):
    # Random weights test executable interfaces only. They do not promise decoding.
    sender_only = tmp_path / "sender-only"
    sender_only.mkdir()
    receiver_only = tmp_path / "receiver-only"
    receiver_only.mkdir()
    config = deepcopy(test_config)
    models = {name: build_component(name, profile) for name in ("ec", "ew", "dc", "dw")}
    paths = {}
    for name, model in models.items():
        directory = sender_only if name in ("ec", "ew") else receiver_only
        path = directory / (name + ".pt")
        torch.save({"architecture": ARCHITECTURE, "component": name,
                    "state_dict": model.state_dict()}, path)
        config["weights"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        paths[name] = path
    registered = register_profile(config, receiver_only / "profiles", allow_test=True)
    actual_profile = load_profile(registered, allow_test=True)
    for name in models:
        loaded = load_component(name, paths[name], actual_profile)
        assert type(loaded) is type(models[name])
    hospital = Ed25519PrivateKey.generate()
    (receiver_only / "hospital.pem").write_bytes(hospital.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    (receiver_only / "color.key").write_bytes(b"c" * 32)
    (receiver_only / "patient.key").write_bytes(b"m" * 32)
    shared = receiver_only / "shared.png"
    sender = Sender(actual_profile, models["ec"], models["ew"], NonceStore(sender_only / "nonce.sqlite"))
    report = sender.send(torch.rand(1, 3, 256, 256), b"t" * 16, b"c" * 32, b"m" * 32, hospital, shared)
    assert report["authentication"] == "AUTHENTIC"
    with pytest.raises(ValueError, match="checksum"):
        load_component("dc", paths["dw"], actual_profile)
    # Remove sender-only files; the child has no original RGB, residual, Ec/Ew or nonce ledger.
    shutil.rmtree(sender_only)
    del sender, models, hospital
    command = [sys.executable, "-m", "dual_payload.medical.cli", "receive", "--allow-test-profile",
               "--profile", str(registered), "--input", str(shared), "--output", str(receiver_only / "result"),
               "--hospital-public-key", str(receiver_only / "hospital.pem"), "--dw", str(paths["dw"]),
               "--dc", str(paths["dc"]), "--color-key", str(receiver_only / "color.key"),
               "--patient-key", str(receiver_only / "patient.key")]
    child = subprocess.run(command, cwd=receiver_only, text=True, capture_output=True, timeout=120,
                           env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])))
    assert child.returncode == 2, child.stderr  # Untrained extraction should fail cleanly.
    result = json.loads(child.stdout)
    assert result["authentication"] == "AUTHENTIC"
    assert result["color"]["status"] in ("DECODE_FAILED", "DECRYPT_FAILED")
    assert result["patient"]["status"] in ("DECODE_FAILED", "DECRYPT_FAILED")
    assert not (receiver_only / "result" / "patient.token").exists()
    assert not (receiver_only / "result" / "rgb.png").exists()


def test_gray8_ste_is_exact_and_has_finite_identity_gradient():
    x = (torch.randn(3, 1, 256, 256) * 2).requires_grad_()
    result = ste_gray8(x)
    expected = torch.round(x.detach().clamp(0, 1) * 255) / 255
    assert torch.equal(result, expected)
    result.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
