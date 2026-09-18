"""FP32 clean training, deterministic validation, checkpointing and evaluation."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import load_config, validate_config
from .data import (FixedCartesianDataset, ImageFolderDataset, SyntheticDataset,
                   ensure_disjoint, load_rgb_image)
from .losses import CleanLoss
from .metrics import compute_metrics, psnr_per_sample, ssim_per_sample
from .system import DualPayloadSystem
from .transforms import rgb_to_ycbcr


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA PyTorch build or use --device cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The baseline does not use AMP or TF32. Some platform operations may still
    # differ numerically; reproducibility is scoped to the same runtime/device.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_loader(dataset, config: dict, training: bool, epoch: int = 0) -> DataLoader:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(config["seed"] + epoch + (0 if training else 100000))
    return DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=training,
                      num_workers=config["data"]["num_workers"], generator=generator,
                      drop_last=False, persistent_workers=False)


@torch.no_grad()
def validate(model, dataset, config: dict, device: torch.device, criterion=None) -> tuple[dict, dict]:
    model.eval()
    count, totals, preview = 0, {}, None
    for batch in make_loader(dataset, config, training=False):
        rgb, message = batch["rgb"].to(device), batch["message"].to(device)
        output = model(rgb, message)
        values = compute_metrics(output, message)
        if criterion is not None:
            values.update({"loss_" + key: value for key, value in criterion(output, message).items()})
        size = len(rgb)
        for key, value in values.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise RuntimeError(f"Non-finite validation metric: {key}")
            totals[key] = totals.get(key, 0.0) + scalar * size
        count += size
        if preview is None:
            preview = {key: output[key][:1].cpu() for key in ("target_rgb", "x_quantized", "rgb_hat")}
    if count == 0:
        raise ValueError("Validation dataset is empty")
    return {key: value / count for key, value in totals.items()}, preview


def save_preview(preview: dict, path: Path) -> None:
    # Display/export only. Losses and raw metrics never receive these clamped values.
    images = [preview["target_rgb"], preview["x_quantized"].expand(-1, 3, -1, -1), preview["rgb_hat"]]
    strip = torch.cat(images, dim=3)[0].clamp(0, 1)
    array = (strip.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def load_diagnostic_checkpoint(path: str | Path) -> dict:
    """Load the non-resumable watermark-overfit format for evaluation only."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("diagnostic_format_version") != 1
            or checkpoint.get("kind") != "watermark_only_bce_overfit"):
        raise ValueError("Full-system manifest evaluation requires a watermark overfit diagnostic checkpoint")
    for key in ("model", "source_config", "diagnostic_steps", "fit_messages"):
        if key not in checkpoint:
            raise ValueError(f"Diagnostic checkpoint is missing {key}")
    return checkpoint


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_experiment_path(value: str, config_path: str | Path | None) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw]
    if config_path is not None:
        candidates.append(Path(config_path).expanduser().resolve().parent / raw)
    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / raw)
    if raw.parts and raw.parts[0] == "..":
        candidates.append(project_root.joinpath(*raw.parts[1:]))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise ValueError(f"Experiment asset does not exist: {value}")


def _load_joint_experiment(config: dict, config_path: str | Path | None):
    experiment = config["experiment"]
    asset_path = _resolve_experiment_path(experiment["asset_manifest"], config_path)
    image_manifest_path = _resolve_experiment_path(experiment["image_manifest"], config_path)
    message_bank_path = _resolve_experiment_path(experiment["message_bank"], config_path)
    message_metadata_path = _resolve_experiment_path(experiment["message_metadata"], config_path)
    with asset_path.open(encoding="utf-8-sig") as stream:
        asset = json.load(stream)
    with image_manifest_path.open(encoding="utf-8-sig") as stream:
        image_manifest = json.load(stream)
    with message_metadata_path.open(encoding="utf-8-sig") as stream:
        message_metadata = json.load(stream)
    expected = {
        "data_protocol": "fixed_full_cartesian_product",
        "image_count": experiment["image_count"],
        "message_count": experiment["message_count"],
        "message_length": experiment["message_bits"],
        "pair_count": experiment["pair_count"],
        "crop_policy": experiment["crop_policy"],
    }
    for key, value in expected.items():
        if asset.get(key) != value:
            raise ValueError(f"Experiment asset mismatch for {key}: {asset.get(key)!r} != {value!r}")
    if _sha256(image_manifest_path) != asset.get("image_split_sha256"):
        raise ValueError("Image manifest SHA-256 does not match experiment manifest")
    if _sha256(message_bank_path) != asset.get("message_bank_sha256"):
        raise ValueError("Message bank SHA-256 does not match experiment manifest")
    if _sha256(message_bank_path) != message_metadata.get("sha256"):
        raise ValueError("Message bank SHA-256 does not match message metadata")
    raw_images = image_manifest.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != experiment["image_count"]:
        raise ValueError("Image manifest count does not match experiment.image_count")
    image_root = _resolve_experiment_path(config["data"]["train_dir"], config_path)
    image_paths = []
    for entry in raw_images:
        if (not isinstance(entry, dict) or not isinstance(entry.get("relative_filename"), str)
                or entry.get("crop_policy") != experiment["crop_policy"]):
            raise ValueError("Invalid image entry in fixed experiment manifest")
        path = (image_root / entry["relative_filename"]).resolve()
        if not path.is_file():
            raise ValueError(f"Fixed experiment image does not exist: {path}")
        if _sha256(path) != entry.get("file_sha256"):
            raise ValueError(f"Image SHA-256 mismatch: {path}")
        image_paths.append(path)
    messages_array = np.load(message_bank_path, allow_pickle=False)
    if (messages_array.shape != (experiment["message_count"], experiment["message_bits"])
            or messages_array.dtype != np.uint8
            or not bool(np.logical_or(messages_array == 0, messages_array == 1).all())):
        raise ValueError("Fixed message bank shape, dtype, or binary values are invalid")
    if message_metadata.get("shape") != list(messages_array.shape):
        raise ValueError("Message metadata shape does not match message bank")
    messages = torch.from_numpy(messages_array.copy()).float()
    dataset = FixedCartesianDataset(image_paths, messages, config["data"]["image_size"])
    if len(dataset) != experiment["pair_count"]:
        raise ValueError("Constructed pair count does not match experiment.pair_count")
    manifest = {
        "experiment": experiment["mode"],
        "train": [path.name for path in image_paths],
        "val": [path.name for path in image_paths],
        "asset_manifest": experiment["asset_manifest"],
    }
    return dataset, manifest


def log_event(path: Path, event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def _metric_summary(values: list[float], name: str) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Invalid full-system metric values: {name}")
    return {"mean": math.fsum(values) / len(values), "min": min(values), "max": max(values)}


@torch.no_grad()
def evaluate_full_system_bank(model, rgbs: list[torch.Tensor], messages: torch.Tensor,
                              device: torch.device, batch_size: int) -> dict:
    """Evaluate the exact image/message Cartesian bank without channel attacks or updates."""
    if not rgbs:
        raise ValueError("Full-system bank needs at least one image")
    if (messages.ndim != 2 or messages.shape[1] != 64 or len(messages) < 1
            or not bool(((messages == 0) | (messages == 1)).all())):
        raise ValueError("Full-system messages must be a non-empty Nx64 binary tensor")
    if batch_size < 1:
        raise ValueError("Full-system batch size must be >= 1")
    model.eval()
    color_psnr, color_ssim = [], []
    full_psnr, full_ssim = [], []
    delta_psnr, delta_ssim = [], []
    carrier_psnr, carrier_ssim = [], []
    bit_errors = complete_messages = total_bits = pairs = 0
    residual_square_sum = residual_elements = 0
    max_absolute_residual = 0.0

    for rgb_cpu in rgbs:
        rgb = rgb_cpu.unsqueeze(0).to(device) if rgb_cpu.ndim == 3 else rgb_cpu.to(device)
        if rgb.shape[0] != 1 or rgb.shape[1] != 3:
            raise ValueError("Each full-system RGB input must have shape 3xHxW or 1x3xHxW")
        y, cb, cr = rgb_to_ycbcr(rgb)
        carrier = model.color_encoder(y, cb, cr)["carrier"]
        color_rgb = model.color_decoder(carrier)["rgb"]
        image_color_psnr = float(psnr_per_sample(color_rgb, rgb)[0])
        image_color_ssim = float(ssim_per_sample(color_rgb, rgb)[0])

        for start in range(0, len(messages), batch_size):
            message = messages[start:start + batch_size].to(device)
            count = len(message)
            host = carrier.expand(count, -1, -1, -1)
            target_rgb = rgb.expand(count, -1, -1, -1)
            watermarked = model.watermark_encoder(host, message)["carrier"]
            logits = model.watermark_decoder(watermarked)
            recovered_rgb = model.color_decoder(watermarked)["rgb"]

            errors = (logits >= 0) != message.bool()
            bit_errors += int(errors.sum())
            complete_messages += int((~errors.any(dim=1)).sum())
            total_bits += errors.numel()
            pairs += count

            batch_full_psnr = [float(value) for value in
                               psnr_per_sample(recovered_rgb, target_rgb).cpu()]
            batch_full_ssim = [float(value) for value in
                               ssim_per_sample(recovered_rgb, target_rgb).cpu()]
            full_psnr.extend(batch_full_psnr)
            full_ssim.extend(batch_full_ssim)
            color_psnr.extend([image_color_psnr] * count)
            color_ssim.extend([image_color_ssim] * count)
            delta_psnr.extend(value - image_color_psnr for value in batch_full_psnr)
            delta_ssim.extend(value - image_color_ssim for value in batch_full_ssim)
            carrier_psnr.extend(float(value) for value in
                                psnr_per_sample(watermarked, host).cpu())
            carrier_ssim.extend(float(value) for value in
                                ssim_per_sample(watermarked, host).cpu())
            residual = watermarked - host
            residual_square_sum += float(residual.double().square().sum())
            residual_elements += residual.numel()
            max_absolute_residual = max(max_absolute_residual, float(residual.abs().max()))

    expected_pairs = len(rgbs) * len(messages)
    if pairs != expected_pairs or total_bits != expected_pairs * 64:
        raise RuntimeError("Full-system Cartesian bank accounting mismatch")
    return {
        "pairs": pairs,
        "watermark": {"ber": bit_errors / total_bits, "bit_errors": bit_errors,
                      "bits": total_bits, "message_accuracy": complete_messages / pairs,
                      "complete_messages": complete_messages},
        "color_only": {"psnr": _metric_summary(color_psnr, "color PSNR"),
                       "ssim": _metric_summary(color_ssim, "color SSIM")},
        "full": {"psnr": _metric_summary(full_psnr, "full PSNR"),
                 "ssim": _metric_summary(full_ssim, "full SSIM")},
        "impact": {"delta_psnr_mean": math.fsum(delta_psnr) / pairs,
                   "delta_ssim_mean": math.fsum(delta_ssim) / pairs},
        "carrier": {"psnr": _metric_summary(carrier_psnr, "carrier PSNR"),
                    "ssim": _metric_summary(carrier_ssim, "carrier SSIM"),
                    "residual_rms": math.sqrt(residual_square_sum / residual_elements),
                    "max_abs": max_absolute_residual},
    }


def _load_full_system_manifest(path: str | Path, checkpoint: dict) -> tuple[dict, list[Path], torch.Tensor]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("mode") != "overfit":
        raise ValueError("Full-system manifest must describe an overfit diagnostic run")
    if manifest.get("crop") != "fixed_center":
        raise ValueError("Full-system manifest must use fixed_center preprocessing")
    config = manifest.get("source_config")
    if not isinstance(config, dict) or config != checkpoint["source_config"]:
        raise ValueError("Manifest source_config does not match the diagnostic checkpoint")
    image_size = manifest.get("image_size")
    if (isinstance(image_size, bool) or not isinstance(image_size, int)
            or image_size < 8 or image_size % 8):
        raise ValueError("Manifest image_size must be a positive multiple of 8")
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list) or not raw_images or not all(isinstance(x, str) for x in raw_images):
        raise ValueError("Manifest images must be a non-empty ordered path list")
    images = [Path(value).expanduser().resolve() for value in raw_images]
    missing = [image_path for image_path in images if not image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest image does not exist: {missing[0]}")
    try:
        messages = torch.tensor(manifest["fit_messages"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Manifest fit_messages must be a rectangular binary list") from error
    if (messages.ndim != 2 or messages.shape[1] != 64 or len(messages) < 1
            or not bool(((messages == 0) | (messages == 1)).all())):
        raise ValueError("Manifest fit_messages must be a non-empty Nx64 binary bank")
    if manifest.get("fit_message_count", len(messages)) != len(messages):
        raise ValueError("Manifest fit_message_count does not match fit_messages")
    saved_messages = checkpoint["fit_messages"].detach().cpu().float()
    if not torch.equal(messages, saved_messages):
        raise ValueError("Manifest fit_messages do not match the diagnostic checkpoint")
    cli_steps = manifest.get("cli", {}).get("steps")
    if cli_steps is not None and cli_steps != checkpoint["diagnostic_steps"]:
        raise ValueError("Manifest steps do not match the diagnostic checkpoint")
    exposure_counts = checkpoint.get("pair_exposure_counts")
    if exposure_counts is not None and exposure_counts.numel() != len(images) * len(messages):
        raise ValueError("Diagnostic checkpoint pair bank size does not match the manifest")
    return manifest, images, messages


def _format_summary(summary: dict[str, float]) -> str:
    return (f"{summary['mean']:.10g} / {summary['min']:.10g} / "
            f"{summary['max']:.10g}")


def print_full_system_report(result: dict) -> None:
    watermark = result["watermark"]
    print("Watermark:")
    print(f"BER = {watermark['ber']:.10g}")
    print(f"bit_errors = {watermark['bit_errors']} / {watermark['bits']}")
    print(f"message_accuracy = {watermark['message_accuracy']:.10g}")
    print("\nColor-only:")
    print(f"PSNR mean/min/max = {_format_summary(result['color_only']['psnr'])}")
    print(f"SSIM mean/min/max = {_format_summary(result['color_only']['ssim'])}")
    print("\nFull:")
    print(f"PSNR mean/min/max = {_format_summary(result['full']['psnr'])}")
    print(f"SSIM mean/min/max = {_format_summary(result['full']['ssim'])}")
    print("\nImpact:")
    print(f"Delta PSNR = {result['impact']['delta_psnr_mean']:.10g}")
    print(f"Delta SSIM = {result['impact']['delta_ssim_mean']:.10g}")
    print("\nCarrier:")
    print(f"PSNR = {result['carrier']['psnr']['mean']:.10g}")
    print(f"SSIM = {result['carrier']['ssim']['mean']:.10g}")
    print(f"residual RMS = {result['carrier']['residual_rms']:.10g}")
    print(f"max abs = {result['carrier']['max_abs']:.10g}", flush=True)


def _check_resume_config(config: dict, saved: dict) -> None:
    for key in ("seed", "model", "channel", "loss", "data", "experiment"):
        if config.get(key) != saved.get(key):
            raise ValueError(f"--resume cannot change {key}; use --init-from for a new experiment")
    for key in ("lr", "weight_decay", "grad_clip"):
        if config["train"][key] != saved["train"][key]:
            raise ValueError(f"--resume cannot change train.{key}; use --init-from")


def train_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the four-network clean baseline (no attacks)")
    parser.add_argument("--config")
    parser.add_argument("--train-dir")
    parser.add_argument("--val-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        help="Split each logical batch into this many micro-batches; default: 1")
    parser.add_argument("--smoke", action="store_true", help="Explicit synthetic data: not a performance experiment")
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument("--resume", help="Restore optimizer and progress in the same run")
    restore.add_argument("--init-from", help="Load model weights into a NEW run, e.g. float -> ste8")
    args = parser.parse_args(argv)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    saved_accumulation_steps = (checkpoint.get("gradient_accumulation_steps", 1)
                                if checkpoint else 1)
    if (checkpoint and args.gradient_accumulation_steps is not None
            and args.gradient_accumulation_steps != saved_accumulation_steps):
        raise ValueError("--resume cannot change gradient accumulation steps")
    accumulation_steps = (saved_accumulation_steps if checkpoint else
                          (args.gradient_accumulation_steps
                           if args.gradient_accumulation_steps is not None else 1))
    if accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    config = deepcopy(checkpoint["config"]) if checkpoint and not args.config else load_config(args.config)
    synthetic = args.smoke or bool(checkpoint and checkpoint["synthetic"])
    fixed_experiment = config.get("experiment") is not None
    if synthetic and fixed_experiment:
        parser.error("--smoke uses synthetic data and cannot be combined with a fixed experiment protocol")
    if args.smoke and (args.train_dir or args.val_dir):
        parser.error("--smoke cannot be combined with real data directories")
    if args.smoke and not checkpoint:
        config["data"].update(image_size=32, batch_size=2, num_workers=0, train_dir=None, val_dir=None)
        config["train"].update(epochs=1, max_steps=3, log_every=1)
    for arg, section, key in (
        (args.train_dir, "data", "train_dir"), (args.val_dir, "data", "val_dir"),
        (args.image_size, "data", "image_size"), (args.batch_size, "data", "batch_size"),
        (args.output_dir, "train", "output_dir"), (args.max_steps, "train", "max_steps"),
        (args.epochs, "train", "epochs"),
    ):
        if arg is not None:
            config[section][key] = arg
    if args.device:
        config["device"] = args.device
    validate_config(config)
    if accumulation_steps > config["data"]["batch_size"]:
        parser.error("--gradient-accumulation-steps cannot exceed the logical batch size")
    if config["channel"]["quantization_mode"] == "real8":
        parser.error("Training requires none or ste8; real8 is evaluation-only")
    if (not synthetic and not fixed_experiment
            and not (config["data"]["train_dir"] and config["data"]["val_dir"])):
        parser.error("Provide independent --train-dir and --val-dir, or explicitly use --smoke")
    if checkpoint:
        _check_resume_config(config, checkpoint["config"])
    device = resolve_device(config["device"])
    seed_everything(config["seed"])
    data = config["data"]
    if fixed_experiment:
        train_data, manifest = _load_joint_experiment(config, args.config)
        val_data = train_data
    elif synthetic:
        train_data = SyntheticDataset(8, data["image_size"], config["seed"])
        val_data = SyntheticDataset(4, data["image_size"], config["seed"] + 10000)
    else:
        train_data = ImageFolderDataset(data["train_dir"], data["image_size"], True, config["seed"])
        val_data = ImageFolderDataset(data["val_dir"], data["image_size"], False, config["seed"] + 10000)
        ensure_disjoint(train_data.paths, val_data.paths)
    # Save the exact ordered file list. A resume must not silently use a changed dataset.
    if not fixed_experiment:
        manifest = {"train": [str(path) for path in getattr(train_data, "paths", [])],
                    "val": [str(path) for path in getattr(val_data, "paths", [])]}
    if checkpoint and manifest != checkpoint["manifest"]:
        raise ValueError("Dataset file list changed since checkpoint")
    model = DualPayloadSystem(config["model"], config["channel"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"],
                                 weight_decay=config["train"]["weight_decay"])
    criterion = CleanLoss(config["loss"])
    global_step, start_epoch, start_batch, best_loss = 0, 0, 0, None
    best_message_ber, best_exact_success = None, None
    if args.init_from:
        initial = load_checkpoint(args.init_from)
        # Budgets affect fixed operations rather than weights: require explicit matching
        # backbone only, allowing experiments with different delta/channel/loss settings.
        for key in ("channels", "blocks"):
            if config["model"][key] != initial["config"]["model"][key]:
                raise ValueError(f"--init-from backbone mismatch: {key}")
        model.load_state_dict(initial["model"], strict=True)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step, start_epoch, start_batch = checkpoint["global_step"], checkpoint["next_epoch"], checkpoint["next_batch"]
        best_loss = checkpoint.get("best_loss")
        best_message_ber = checkpoint.get("best_message_ber")
        best_exact_success = checkpoint.get("best_exact_success")
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if start_epoch >= config["train"]["epochs"]:
        raise ValueError("No remaining epochs; increase --epochs to resume")
    maximum = config["train"]["max_steps"]
    if maximum is not None and global_step >= maximum:
        raise ValueError("--max-steps must exceed the checkpoint global_step")
    if checkpoint:
        run_dir = Path(args.resume).resolve().parent
        requested = config["train"]["output_dir"]
        if requested and Path(requested).resolve() != run_dir:
            raise ValueError("--resume must use its existing run directory; use --init-from for a new run")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = config["channel"]["quantization_mode"]
        run_dir = Path(config["train"]["output_dir"] or f"runs/{'smoke' if synthetic else 'clean'}_{mode}_{stamp}").resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    config["train"]["output_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "manifest.json", manifest)
    log_path = run_dir / "metrics.jsonl"
    start_event = {"kind": "resume" if checkpoint else "start", "synthetic": synthetic,
                   "device": str(device), "torch": str(torch.__version__),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "train_images": len(train_data), "val_images": len(val_data),
                   "global_step": global_step, "output_dir": str(run_dir)}
    if accumulation_steps != 1:
        start_event["gradient_accumulation_steps"] = accumulation_steps
    log_event(log_path, start_event)
    for epoch in range(start_epoch, config["train"]["epochs"]):
        loader = make_loader(train_data, config, training=True, epoch=epoch)
        model.train()
        next_epoch, next_batch = epoch + 1, 0
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            logical_size = len(batch["rgb"])
            if accumulation_steps == 1:
                rgb = batch["rgb"].to(device)
                if fixed_experiment:
                    message = batch["message"].to(device)
                else:
                    # Generic training uses fresh messages independent of image index.
                    message = torch.randint(0, 2, (len(rgb), 64), device=device).float()
                output = model(rgb, message)
                losses = criterion(output, message)
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError("Non-finite loss; stop rather than save invalid weights")
                losses["total"].backward()
                logged_losses = {key: float(value.detach()) for key, value in losses.items()}
            else:
                logical_messages = (batch["message"] if fixed_experiment else
                                    torch.randint(0, 2, (logical_size, 64),
                                                  device=device).float())
                logged_losses = {}
                for micro_index in range(accumulation_steps):
                    start = micro_index * logical_size // accumulation_steps
                    end = (micro_index + 1) * logical_size // accumulation_steps
                    if start == end:
                        continue
                    micro_rgb = batch["rgb"][start:end].to(device)
                    micro_message = (logical_messages[start:end].to(device) if fixed_experiment
                                     else logical_messages[start:end])
                    micro_output = model(micro_rgb, micro_message)
                    micro_losses = criterion(micro_output, micro_message)
                    if not bool(torch.isfinite(micro_losses["total"])):
                        raise RuntimeError("Non-finite loss; stop rather than save invalid weights")
                    scale = (end - start) / logical_size
                    (micro_losses["total"] * scale).backward()
                    for key, value in micro_losses.items():
                        logged_losses[key] = logged_losses.get(key, 0.0) + float(value.detach()) * scale
                    del micro_rgb, micro_message, micro_output, micro_losses
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            global_step += 1
            if global_step % config["train"]["log_every"] == 0 or global_step == 1:
                log_event(log_path, {"kind": "train", "epoch": epoch + 1, "global_step": global_step,
                                     **{"loss_" + key: value for key, value in logged_losses.items()}})
            if maximum is not None and global_step >= maximum:
                if batch_index + 1 < len(loader):
                    next_epoch, next_batch = epoch, batch_index + 1
                break
        metrics, preview = validate(model, val_data, config, device, criterion)
        log_event(log_path, {"kind": "validation", "epoch": epoch + 1, "global_step": global_step, **metrics})
        if fixed_experiment:
            improved_ber = best_message_ber is None or metrics["ber"] < best_message_ber
            improved_exact = (best_exact_success is None
                              or metrics["message_accuracy"] > best_exact_success)
            if improved_ber:
                best_message_ber = metrics["ber"]
            if improved_exact:
                best_exact_success = metrics["message_accuracy"]
            improved = False
        else:
            improved = best_loss is None or metrics["loss_total"] < best_loss
            if improved:
                best_loss = metrics["loss_total"]
        state = {"format_version": 1, "config": deepcopy(config), "synthetic": synthetic,
                 "manifest": manifest, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "global_step": global_step, "next_epoch": next_epoch, "next_batch": next_batch,
                 "best_loss": best_loss, "best_message_ber": best_message_ber,
                 "best_exact_success": best_exact_success, "validation": metrics,
                 "gradient_accumulation_steps": accumulation_steps,
                 "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                 "torch_version": str(torch.__version__)}
        save_checkpoint(run_dir / "last.pt", state)
        if fixed_experiment and improved_ber:
            save_checkpoint(run_dir / "best_message_ber.pt", state)
        if fixed_experiment and improved_exact:
            save_checkpoint(run_dir / "best_exact_success.pt", state)
        if not fixed_experiment and improved:
            save_checkpoint(run_dir / "best.pt", state)
        save_preview(preview, run_dir / "preview.png")
        if maximum is not None and global_step >= maximum:
            break
    print(f"Completed. Checkpoint: {run_dir / 'last.pt'}", flush=True)


def evaluate_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate clean float or real8 images with fixed messages")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--manifest", help="Exact watermark-overfit image/message bank for full-system evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quantization-mode", choices=("none", "real8"))
    parser.add_argument("--batch-size", type=int, help="Full-system pair batch size; default: manifest eval batch size")
    parser.add_argument("--output", help="Optional new JSON output file")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if sum((bool(args.smoke), bool(args.data_dir), bool(args.manifest))) != 1:
        parser.error("Choose exactly one of --data-dir DIR, --manifest FILE, or explicit --smoke")
    if args.manifest and args.quantization_mode:
        parser.error("--quantization-mode is not used with the exact full-system manifest flow")
    if not args.manifest and args.batch_size is not None:
        parser.error("--batch-size is only available with --manifest")
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    if output_path is not None and output_path.exists():
        raise ValueError(f"Output already exists: {output_path}")

    if args.manifest:
        checkpoint = load_diagnostic_checkpoint(args.checkpoint)
        manifest, image_paths, messages = _load_full_system_manifest(args.manifest, checkpoint)
        config = deepcopy(checkpoint["source_config"])
        if config["channel"] != {"quantization_mode": "none", "clamp_enabled": False,
                                "attack_mode": "identity"}:
            raise ValueError("Full-system baseline requires the saved float identity configuration")
        batch_size = args.batch_size or manifest.get("eval_batch_size") or config["data"]["batch_size"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Full-system batch size must be an integer >= 1")
        seed_everything(config["seed"])
        device = resolve_device(args.device)
        model = DualPayloadSystem(config["model"], config["channel"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        rgbs = [load_rgb_image(path, manifest["image_size"]) for path in image_paths]
        result = evaluate_full_system_bank(model, rgbs, messages, device, batch_size)
        result.update({
            "mode": "full_system_baseline",
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_kind": checkpoint["kind"],
            "diagnostic_steps": checkpoint["diagnostic_steps"],
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "images": [str(path) for path in image_paths],
            "image_count": len(image_paths), "message_count": len(messages),
            "messages": messages.int().tolist(), "pair_order": "image_major_cartesian",
            "preprocessing": {"image_size": manifest["image_size"], "crop": manifest["crop"]},
            "device": str(device), "torch": str(torch.__version__),
        })
        print_full_system_report(result)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, result)
            print(f"Report: {output_path}", flush=True)
        return

    checkpoint = load_checkpoint(args.checkpoint)
    config = deepcopy(checkpoint["config"])
    if args.quantization_mode:
        config["channel"].update(quantization_mode=args.quantization_mode,
                                  clamp_enabled=args.quantization_mode == "real8")
    elif config["channel"]["quantization_mode"] == "ste8":
        config["channel"]["quantization_mode"] = "real8"
    seed_everything(config["seed"])
    device = resolve_device(args.device)
    model = DualPayloadSystem(config["model"], config["channel"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = (SyntheticDataset(4, config["data"]["image_size"], config["seed"] + 10000)
               if args.smoke else ImageFolderDataset(args.data_dir, config["data"]["image_size"],
                                                     False, config["seed"] + 10000))
    metrics, _ = validate(model, dataset, config, device)
    result = {"synthetic": args.smoke, "images": len(dataset),
              "channel": config["channel"], "metrics": metrics}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, result)


def run_cli(main) -> None:
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
