"""Command line file sender/receiver and public profile registration."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from PIL import Image

from .crypto import NonceStore
from .models import load_component
from .pipeline import Receiver, Sender
from .png import hospital_key_id
from .preprocess import integer_pixels, prepare_work_image
from .profile import load_profile, read_json, register_profile


def _secret(path, size):
    data = Path(path).read_bytes()
    if len(data) != size:
        raise ValueError(f"{path}: expected {size} raw bytes")
    return data


def _pem(path, private=False):
    raw = Path(path).read_bytes()
    key = (serialization.load_pem_private_key(raw, password=None) if private
           else serialization.load_pem_public_key(raw))
    if not isinstance(key, Ed25519PrivateKey if private else Ed25519PublicKey):
        raise ValueError("Hospital signing keys must use Ed25519 PEM")
    return key


def parser():
    parser = argparse.ArgumentParser(description="Medical V1 authenticated grayscale sharing")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register-profile")
    register.add_argument("--config", required=True)
    register.add_argument("--registry", required=True)
    register.add_argument("--allow-test-profile", action="store_true")
    for name in ("send", "receive"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--profile", required=True, help="Trusted registered profile directory")
        cmd.add_argument("--allow-test-profile", action="store_true")
        cmd.add_argument("--device", default="cpu", help="cpu or cuda:N (FP32)")
        cmd.add_argument("--threads", type=int, default=1)
        cmd.add_argument("--input", required=True)
        cmd.add_argument("--output", required=True, help="New PNG (send) or new output directory (receive)")
        cmd.add_argument("--color-key", required=name == "send", help="External file with 32 raw bytes")
        cmd.add_argument("--patient-key", required=name == "send", help="External file with 32 raw bytes")
        if name == "send":
            cmd.add_argument("--ec", required=True)
            cmd.add_argument("--ew", required=True)
            cmd.add_argument("--token", required=True, help="External file with 16 raw bytes")
            cmd.add_argument("--hospital-private-key", required=True)
            cmd.add_argument("--nonce-store", required=True, help="Persistent shared SQLite ledger")
        else:
            cmd.add_argument("--diagnostics", action="store_true", help="Save receiver-only logits and LDPC bits for evaluation")
            cmd.add_argument("--dw", help="Trusted local Dw checkpoint")
            cmd.add_argument("--dc", help="Trusted local Dc checkpoint; only needed for color")
            cmd.add_argument("--hospital-public-key", action="append", default=[],
                             help="Trusted Ed25519 PEM; repeat for multiple hospitals")
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "register-profile":
        directory = register_profile(read_json(args.config), args.registry,
                                     allow_test=args.allow_test_profile)
        print(json.dumps({"profile": str(directory.resolve())}))
        return 0
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    profile = load_profile(args.profile, allow_test=args.allow_test_profile)
    color_key = _secret(args.color_key, 32) if args.color_key else None
    patient_key = _secret(args.patient_key, 32) if args.patient_key else None
    if args.command == "send":
        rgb, content_rect = prepare_work_image(args.input)
        sender = Sender(profile, load_component("ec", args.ec, profile, args.device),
                        load_component("ew", args.ew, profile, args.device),
                        NonceStore(args.nonce_store), device=args.device)
        report = sender.send(rgb, _secret(args.token, 16), color_key, patient_key,
                             _pem(args.hospital_private_key, private=True), args.output, content_rect)
        print(json.dumps(report, sort_keys=True))
        return 0
    trusted = {}
    for path in args.hospital_public_key:
        public_key = _pem(path)
        trusted[hospital_key_id(public_key)] = public_key

    def loader(name):
        path = getattr(args, name)
        if path is None:
            raise ValueError(f"--{name} is required for this permission")
        return load_component(name, path, profile, args.device)

    receiver = Receiver(profile, trusted, loader, device=args.device)
    diagnostics = {}
    def observe(event, value):
        if event == 'logits':
            diagnostics.update({key: tensor.cpu().numpy() for key, tensor in value.items()})
        else:
            diagnostics[event] = value
    result = receiver.receive(args.input, color_key, patient_key,
                              observer=observe if args.diagnostics else None)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    if args.diagnostics:
        np.savez_compressed(destination / "diagnostics.npz", **diagnostics)
    if result.color.rgb is not None:
        np.save(destination / "rgb_float.npy", result.color.rgb, allow_pickle=False)
        rgb8 = integer_pixels(torch.from_numpy(result.color.rgb)).transpose(1, 2, 0)
        Image.fromarray(rgb8).save(destination / "rgb.png")
    if result.patient.token is not None:
        with (destination / "patient.token").open("xb") as stream:
            stream.write(result.patient.token)
    report = result.summary()
    (destination / "status.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    success = result.authentication == "AUTHENTIC" and all(
        branch.status in ("OK", "KEY_MISSING") for branch in (result.color, result.patient))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
