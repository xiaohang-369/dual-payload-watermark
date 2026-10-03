"""Medical V3 joint training from scratch and frozen Protocol evaluation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .checkpoints import save_checkpoint
from .config import load_config, validate_config
from .crypto import require_bytes
from .data import ManifestDataset, SyntheticDataset, ensure_disjoint
from .keyed_permutation import unpack_selectors
from .losses import CleanLoss
from .metrics import compute_metrics, psnr, ssim
from .package import verify_package
from .protocol import ProtocolV1, fp32_inference
from .stages import apply_training_stage, build_optimizer, trainable_parameters
from .system import DualPayloadSystem
from .transforms import rgb_to_ycbcr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Current validated numerical paths are CPU/CUDA FP32")
    return device


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    # Serialize before opening so invalid numbers cannot leave a partial JSON line.
    line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def make_overfit8_dataset(manifest, seed):
    if not manifest:
        raise ValueError("--overfit8 requires a prepared train_manifest")
    dataset = ManifestDataset(manifest, training=False, seed=seed)
    if len(dataset) < 8:
        raise ValueError("--overfit8 requires at least 8 manifest samples")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), 8))
    # Subset preserves original row indices for ManifestDataset.fixed_message.
    selection = {"seed": seed, "manifest": str(dataset.manifest), "samples": [
        {"original_manifest_row_index": index, "path": dataset.rows[index]["path"],
         **{key: dataset.rows[index][key] for key in ("patient_id", "img_id") if key in dataset.rows[index]}}
        for index in indices]}
    return Subset(dataset, indices), selection


def make_loader(dataset, config, training=False, epoch=0):
    return DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=training,
                      num_workers=config["data"]["num_workers"],
                      generator=torch.Generator().manual_seed(config["seed"] + epoch))


@torch.no_grad()
def validate(model, dataset, config, device, criterion=None):
    model.eval()
    totals, count = {}, 0
    for batch in make_loader(dataset, config):
        rgb, message = batch["rgb"].to(device), batch["message"].to(device)
        output = model(rgb, message)
        values = compute_metrics(output, message)
        if criterion is not None:
            values.update({f"loss_{key}": value for key, value in criterion(output, message).items()})
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + float(value) * rgb.shape[0]
        count += rgb.shape[0]
    if count == 0:
        raise ValueError("Validation dataset is empty")
    return {key: value / count for key, value in totals.items()}


def train_step(model, batch, config, optimizer, criterion, device):
    if config["train"]["stage"] == "protocol_eval":
        raise ValueError("protocol_eval cannot perform a training step")
    apply_training_stage(model, config)
    optimizer.zero_grad(set_to_none=True)
    message = batch["message"].to(device)
    output = model(batch["rgb"].to(device), message)
    losses = criterion(output, message)
    if not bool(torch.isfinite(losses["total"])):
        raise ValueError("Nonfinite training loss")
    losses["total"].backward()  # Exactly one backward for the complete chain.
    torch.nn.utils.clip_grad_norm_(trainable_parameters(model), config["train"]["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {key: float(value.detach()) for key, value in losses.items()}


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Medical V3 joint_256 training from scratch")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Explicit synthetic data only")
    mode.add_argument("--overfit8", action="store_true", help="Fit 8 fixed prepared image-message pairs; engineering sanity check only")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config["train"]["stage"] == "protocol_eval":
        raise ValueError("Use evaluate.py for protocol_eval; training is prohibited")
    if args.device:
        config["device"] = args.device
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    validate_config(config)
    output_dir = config["train"]["output_dir"]
    if not output_dir:
        raise ValueError("Set train.output_dir or --output-dir explicitly")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty")
    if args.smoke:
        train_data = SyntheticDataset(2, seed=config["seed"], training=True)
        val_data = SyntheticDataset(1, seed=config["seed"] + 10000)
    elif args.overfit8:
        train_data, selection = make_overfit8_dataset(config["data"]["train_manifest"], config["seed"])
        val_data = train_data  # Same fixed pairs only in this explicit sanity-check mode.
    else:
        paths = (config["data"]["train_manifest"], config["data"]["val_manifest"])
        if not all(paths):
            raise ValueError("Prepared train_manifest and val_manifest are required; no implicit synthetic data")
        train_data = ManifestDataset(paths[0], training=True, seed=config["seed"])
        val_data = ManifestDataset(paths[1], seed=config["seed"] + 10000)
        ensure_disjoint(train_data.paths, val_data.paths)
        train_ids = {row["patient_id"] for row in train_data.rows if row.get("patient_id") is not None}
        val_ids = {row["patient_id"] for row in val_data.rows if row.get("patient_id") is not None}
        if train_ids & val_ids:
            raise ValueError("Supplied train/validation patient IDs overlap")
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    model = DualPayloadSystem(config["model"], config["channel"]).to(device)
    optimizer = build_optimizer(model, config)
    criterion = CleanLoss(config["loss"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    if args.overfit8:
        write_json(output_dir / "overfit8_selection.json", selection)
    global_step, best_ber, best_loss = 0, float("inf"), float("inf")
    for epoch in range(config["train"]["epochs"]):
        train_totals, train_samples = {}, 0
        for batch in make_loader(train_data, config, training=True, epoch=epoch):
            losses = train_step(model, batch, config, optimizer, criterion, device)
            batch_size = batch["rgb"].shape[0]
            for key, value in losses.items():
                train_totals[key] = train_totals.get(key, 0.) + value * batch_size
            train_samples += batch_size
            global_step += 1
            if global_step == 1 or global_step % config["train"]["log_every"] == 0:
                print(json.dumps({"stage": config["train"]["stage"], "step": global_step, "loss": losses}), flush=True)
            if config["train"]["max_steps"] is not None and global_step >= config["train"]["max_steps"]:
                break
        metrics = validate(model, val_data, config, device, criterion)
        state = {"optimizer": optimizer.state_dict(), "global_step": global_step,
                 "epoch": epoch, "validation": metrics, "synthetic": args.smoke,
                 "overfit8": args.overfit8}
        save_checkpoint(output_dir / "last.pt", model, deepcopy(config), **state)
        if metrics["ber"] < best_ber:
            save_checkpoint(output_dir / "best_message_ber.pt", model, deepcopy(config), **state)
            best_ber = metrics["ber"]
        if metrics["loss_total"] < best_loss:
            save_checkpoint(output_dir / "best.pt", model, deepcopy(config), **state)
            best_loss = metrics["loss_total"]
        write_json(output_dir / "validation.json", metrics)
        append_jsonl(output_dir / "metrics.jsonl", {
            "epoch": epoch, "global_step": global_step, "train_samples": train_samples,
            "train_loss": {key: total / train_samples for key, total in train_totals.items()},
            "validation": metrics, "mode": "overfit8" if args.overfit8 else "joint_256"})
        if config["train"]["max_steps"] is not None and global_step >= config["train"]["max_steps"]:
            break
    print(json.dumps({"output_dir": str(output_dir), "steps": global_step, "synthetic": args.smoke}), flush=True)


def evaluate_protocol_sample(engine, rgb, token, kc, km, signing_key, parameters):
    data, expected = engine._publish_for_evaluation(rgb, token, kc, km, signing_key, parameters)
    public_key = signing_key.public_key()
    verified = verify_package(data, public_key, expected_model_id=engine.model_id)
    restored = engine.recover_color(data, kc, public_key)
    try:
        authenticated = engine.recover_patient(data, km, public_key) == token
    except InvalidTag:
        authenticated = False  # Keep failure in metrics; never release candidate plaintext.
    with fp32_inference(engine.device):
        rgb = rgb.to(engine.device)
        baseline = engine.model(rgb, expected)
        logits = engine._patient_logits(verified, km)
        errors = (logits >= 0) != expected[0].bool()
        y = rgb_to_ycbcr(rgb)[0]
        gray = verified.gray.to(engine.device)
        metrics = {"ber": float(errors.float().mean()), "exact_256_bits": not bool(errors.any()),
                   "patient_authenticated": authenticated, "gray_psnr": float(psnr(gray, y)),
                   "gray_ssim": float(ssim(gray, y)), "authorized_rgb_psnr": float(psnr(restored, rgb)),
                   "authorized_rgb_ssim": float(ssim(restored, rgb)),
                   "baseline_rgb_psnr": float(psnr(baseline["rgb_hat"], rgb)),
                   "authorized_vs_baseline_rms": float((restored - baseline["rgb_hat"]).square().mean().sqrt()),
                   "permutation_added_rms": float((gray - baseline["x_float"]).square().mean().sqrt()),
                   "gray_oob_fraction": float(((gray < 0) | (gray > 1)).float().mean())}
    c, m = unpack_selectors(verified.raw.selectors)
    metrics.update(color_fallback=sum(v == 0 for v in c) / 1024,
                   patient_fallback=sum(v == 0 for v in m) / 1024)
    return data, metrics


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen medical V3 Protocol v1 evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--kc-file", required=True, help="32 raw bytes")
    parser.add_argument("--km-file", required=True, help="32 raw bytes")
    parser.add_argument("--signing-key-file", required=True, help="Ed25519 private seed, 32 raw bytes")
    parser.add_argument("--token-file", required=True, help="16 raw bytes supplied by upstream")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config["train"]["stage"] != "protocol_eval":
        raise ValueError("evaluate.py requires protocol_eval")
    engine = ProtocolV1(args.checkpoint, device=resolve_device(args.device or config["device"]))
    if engine.config["model"] != config["model"]:
        raise ValueError("Evaluation model config differs from the checkpoint")
    apply_training_stage(engine.model, config)
    kc = require_bytes(Path(args.kc_file).read_bytes(), 32, "KC")
    km = require_bytes(Path(args.km_file).read_bytes(), 32, "KM")
    signing_key = Ed25519PrivateKey.from_private_bytes(Path(args.signing_key_file).read_bytes())
    token = require_bytes(Path(args.token_file).read_bytes(), 16, "PatientToken")
    if args.smoke:
        dataset = SyntheticDataset(1, seed=config["seed"])
    else:
        manifest = args.manifest or config["data"]["val_manifest"]
        if not manifest:
            raise ValueError("Supply a prepared manifest or explicitly select --smoke")
        dataset = ManifestDataset(manifest, seed=config["seed"])
    destination = Path(args.output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for i in range(len(dataset)):
        sample = dataset[i]
        package, metrics = evaluate_protocol_sample(engine, sample["rgb"].unsqueeze(0), token,
                                                    kc, km, signing_key, config["protocol"])
        (destination / f"{i:06d}.dpw").write_bytes(package)
        reports.append({"index": i, **metrics})
    write_json(destination / "report.json", {"stage": "protocol_eval", "synthetic": args.smoke,
               "model_id": engine.model_id.hex(), "protocol": config["protocol"], "images": reports})
    print(json.dumps({"images": len(reports), "output_dir": str(destination)}), flush=True)


def run_cli(main):
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
