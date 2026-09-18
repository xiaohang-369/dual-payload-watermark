"""Opt-in clean watermark probes; never resume or overwrite a baseline run."""

import argparse
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import ImageFolderDataset
from .losses import CleanLoss
from .system import DualPayloadSystem
from .training import (load_checkpoint, log_event, resolve_device, seed_everything,
                       write_json)
from .transforms import rgb_to_ycbcr


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def message_metrics(logits, messages, include_per_bit=False):
    errors = (logits >= 0) != messages.bool()
    result = {"bce": float(F.binary_cross_entropy_with_logits(logits, messages)),
              "ber": float(errors.float().mean()),
              "message_accuracy": float((~errors.any(dim=1)).float().mean()),
              "logit_rms": rms(logits)}
    if include_per_bit:
        bit_error_counts = errors.sum(dim=0).detach().cpu()
        result.update({
            "samples": len(messages),
            "bits": int(errors.numel()),
            "bit_errors": int(errors.sum()),
            "per_bit_error_counts": [int(value) for value in bit_error_counts],
            "per_bit_ber": [float(value) for value in bit_error_counts.double() / len(messages)],
        })
    return result


def _draw_unique_messages(count, generator, seen):
    """Draw CPU messages, mutating ``seen`` to prevent train/validation overlap."""
    messages = []
    while len(messages) < count:
        message = torch.randint(0, 2, (64,), generator=generator).float()
        key = tuple(int(value) for value in message.tolist())
        if key not in seen:
            seen.add(key)
            messages.append(message)
    return torch.stack(messages)


def random_message_bank(count, seed):
    """A reproducible bank of distinct CPU messages."""
    if count < 1:
        raise ValueError("Message bank must contain at least one message")
    return _draw_unique_messages(count, torch.Generator().manual_seed(seed), set())


def message_banks(count, seed):
    """Disjoint fixed fit/held-out messages, shared across every selected image."""
    if count < 2:
        raise ValueError("Need at least two messages per image")
    bank = random_message_bank(2 * count, seed)
    return bank[:count], bank[count:]


def fixed_codebook_banks(fit_count, validation_count, fit_seed, validation_seed):
    """Independent fixed fit/validation banks; validation matches random_messages."""
    if min(fit_count, validation_count) < 2:
        raise ValueError("Fixed codebook banks need at least two messages each")
    fit = random_message_bank(fit_count, fit_seed)
    validation = random_message_bank(validation_count, validation_seed)
    fit_keys = {tuple(int(value) for value in message.tolist()) for message in fit}
    validation_keys = {tuple(int(value) for value in message.tolist()) for message in validation}
    if fit_keys.intersection(validation_keys):
        raise RuntimeError("Independent fixed fit/validation message banks unexpectedly overlap")
    return fit, validation


def prior_baseline(fit_messages, heldout_messages):
    """A decoder that ignores the image can exploit a small bank's per-bit imbalance."""
    probability = fit_messages.mean(dim=0, keepdim=True).clamp(1e-6, 1 - 1e-6)
    logits = torch.logit(probability)
    return {"fit_bank": message_metrics(logits.expand(len(fit_messages), -1), fit_messages),
            "heldout_messages": message_metrics(logits.expand(len(heldout_messages), -1),
                                                 heldout_messages),
            "fit_bit_one_counts": [int(value) for value in fit_messages.sum(dim=0)],
            "fit_bit_one_frequency": [float(value) for value in fit_messages.mean(dim=0)]}


@torch.no_grad()
def inspect_image(model, rgb, messages):
    """Sequential messages bound VRAM; all comparisons use exactly the same host."""
    y, cb, cr = rgb_to_ycbcr(rgb)
    s = model.color_encoder(y, cb, cr)["carrier"]
    encoder, decoder = model.watermark_encoder, model.watermark_decoder
    candidates, projected, residuals, logits, embeddings, gains = [], [], [], [], [], []
    for message in messages:
        message = message[None].to(rgb.device)
        water = encoder(s, message)
        projection = encoder.dct.project(water["candidate"], "w")
        gain = (encoder.delta_w / (projection.square().mean() + encoder.eps).sqrt()).clamp(max=1)
        candidates.append(water["candidate"].cpu())
        projected.append(projection.cpu())
        residuals.append(water["residual"].cpu())
        logits.append(decoder(water["carrier"]).cpu())
        embeddings.append(encoder.message_branch(2 * message - 1).cpu())
        gains.append(float(gain))
    candidates, projected, residuals, logits, embeddings = [torch.cat(items) for items in
                                                           (candidates, projected, residuals, logits, embeddings)]
    messages = messages.cpu()
    no_water_logits = decoder(s).cpu().expand(len(messages), -1)
    # One controlled bit flip, explicitly not a survey of all 64 bit positions.
    flipped = messages[:1].clone()
    flipped[:, 0] = 1 - flipped[:, 0]
    flip_water = encoder(s, flipped.to(rgb.device))
    flip_logits = decoder(flip_water["carrier"]).cpu()
    values = {"matched": message_metrics(logits, messages),
              "mismatched_labels": message_metrics(logits, messages.roll(1, dims=0)),
              "watermark_removed": message_metrics(no_water_logits, messages),
              "candidate_rms": rms(candidates), "projected_rms": rms(projected),
              "residual_rms": rms(residuals), "cap_gain_min": min(gains),
              "cap_gain_mean": sum(gains) / len(gains),
              "cap_active_fraction": sum(gain < 1 for gain in gains) / len(gains),
              "host_pw_rms": rms(encoder.dct.project(s, "w")),
              "residual_over_budget": rms(residuals) / encoder.delta_w if encoder.delta_w else None,
              "single_flip_bit_index": 0,
              "single_flip_residual_diff_rms": rms(flip_water["residual"].cpu() - residuals[:1]),
              "single_flip_logits_diff_rms": rms(flip_logits - logits[:1])}
    # Across-message standard deviation, averaged in RMS sense over tensor entries.
    for name, value in (("embedding", embeddings), ("candidate", candidates),
                        ("projected", projected), ("residual", residuals), ("logits", logits)):
        values[name + "_message_variation_rms"] = rms(value - value.mean(dim=0, keepdim=True))
    return values


def gradient_stats(module):
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return {"parameter_tensors": len(parameters), "with_grad_tensors": len(gradients),
            "nonzero_grad_tensors": sum(bool(torch.count_nonzero(gradient)) for gradient in gradients),
            "finite": finite,
            "grad_l2": math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients))
            if finite else None}


def gradient_probe(model, rgb, message, config):
    """Backward only, no optimizer and no clipping/parameter updates. Fresh model assumed."""
    encoder = model.watermark_encoder
    groups = {"color_encoder": model.color_encoder, "color_decoder": model.color_decoder,
              "watermark_encoder": encoder, "message_branch": encoder.message_branch,
              "watermark_fusion": encoder.fusion, "watermark_encoder_body": encoder.body,
              "watermark_encoder_head": encoder.head,
              "watermark_decoder": model.watermark_decoder,
              "watermark_decoder_head": model.watermark_decoder.head}
    result = {}
    for objective in ("message", "total"):
        model.zero_grad(set_to_none=True)
        captured = {}

        def capture(module, inputs, output):
            # Observe actual encoder tensors, without reimplementing its forward path.
            for key in ("candidate", "residual"):
                output[key].retain_grad()
                captured[key] = output[key]

        handle = encoder.register_forward_hook(capture)
        try:
            output = model(rgb, message)
            losses = CleanLoss(config["loss"])(output, message)
            losses[objective].backward()
            stats = {name: gradient_stats(module) for name, module in groups.items()}
            whole = gradient_stats(model)
            norm = whole["grad_l2"]
            result[objective] = {
                "loss": float(losses[objective].detach()), "groups": stats, "all_parameters": whole,
                "hypothetical_global_clip_gain": min(1., config["train"]["grad_clip"] / (norm + 1e-6))
                if norm is not None else None,
                "activation_gradient_rms": {
                    key: rms(value.grad) if value.grad is not None and bool(torch.isfinite(value.grad).all())
                    else None for key, value in captured.items()}}
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def evaluate_bank(model, hosts, messages, device, batch_size=1):
    """Evaluate a fixed bank plus label-mismatch and no-watermark controls."""
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be >= 1")
    model.eval()
    logits, no_watermark_logits = [], []
    for host in hosts:
        host = host.to(device)
        no_watermark = model.watermark_decoder(host).cpu()
        no_watermark_logits.append(no_watermark.expand(len(messages), -1))
        for start in range(0, len(messages), batch_size):
            batch = messages[start:start + batch_size].to(device)
            hosts_batch = host.expand(len(batch), -1, -1, -1)
            water = model.watermark_encoder(hosts_batch, batch)
            logits.append(model.watermark_decoder(water["carrier"]).cpu())
    logits = torch.cat(logits)
    no_watermark_logits = torch.cat(no_watermark_logits)
    targets = messages.repeat(len(hosts), 1)
    return {
        "matched": message_metrics(logits, targets, include_per_bit=True),
        "mismatched_labels": message_metrics(logits, targets.roll(1, dims=0)),
        "watermark_removed": message_metrics(no_watermark_logits, targets),
    }


@torch.no_grad()
def evaluate_random_messages(model, host, messages, device, batch_size):
    """Evaluate matched, mismatched-label and watermark-removed controls on one host."""
    model.eval()
    host = host.to(device)
    logits = []
    for start in range(0, len(messages), batch_size):
        batch = messages[start:start + batch_size].to(device)
        hosts = host.expand(len(batch), -1, -1, -1)
        water = model.watermark_encoder(hosts, batch)
        logits.append(model.watermark_decoder(water["carrier"]).cpu())
    logits = torch.cat(logits)
    no_watermark_logits = model.watermark_decoder(host).cpu().expand(len(messages), -1)
    return {
        "matched": message_metrics(logits, messages, include_per_bit=True),
        "mismatched_labels": message_metrics(logits, messages.roll(1, dims=0)),
        "watermark_removed": message_metrics(no_watermark_logits, messages),
    }


def overfit(model, rgbs, fit_messages, heldout_messages, config, steps, batch_size,
            log_every, device, run_dir, eval_batch_size=None):
    """Watermark-only BCE fit on a Cartesian image/message bank; NOT joint training."""
    model.eval()
    for module in (model.color_encoder, model.color_decoder):
        module.requires_grad_(False)
    with torch.no_grad():
        hosts = [model.color_encoder(*rgb_to_ycbcr(rgb.to(device)))["carrier"].cpu() for rgb in rgbs]
    parameters = list(model.watermark_encoder.parameters()) + list(model.watermark_decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["train"]["lr"],
                                 weight_decay=config["train"]["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"] + 900000)
    pairs = len(hosts) * len(fit_messages)
    order, cursor = torch.randperm(pairs, generator=generator), 0
    exposures = torch.zeros(pairs, dtype=torch.long)
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    if eval_batch_size < 1:
        raise ValueError("Evaluation batch size must be >= 1")
    history = []
    prior = prior_baseline(fit_messages, heldout_messages)

    def evaluate(step):
        fit_evaluation = evaluate_bank(model, hosts, fit_messages, device, eval_batch_size)
        heldout_evaluation = evaluate_bank(model, hosts, heldout_messages, device,
                                           eval_batch_size)
        event = {"kind": "overfit_validation", "diagnostic_step": step,
                 "training_presentations": int(exposures.sum()),
                 "pair_exposure_min": int(exposures.min()),
                 "pair_exposure_max": int(exposures.max()),
                 "pair_exposure_gap": int(exposures.max() - exposures.min()),
                 "pair_exposure_counts": [int(value) for value in exposures],
                 "fit_bank": fit_evaluation["matched"],
                 "heldout_messages_same_images": heldout_evaluation["matched"],
                 "fit_controls": {key: value for key, value in fit_evaluation.items()
                                  if key != "matched"},
                 "heldout_controls": {key: value for key, value in heldout_evaluation.items()
                                      if key != "matched"},
                 "message_prior_without_image": prior}
        history.append(event)
        log_event(run_dir / "metrics.jsonl", event)

    evaluate(0)
    for step in range(1, steps + 1):
        indices = []
        while len(indices) < batch_size:
            if cursor == pairs:
                order, cursor = torch.randperm(pairs, generator=generator), 0
            take = min(batch_size - len(indices), pairs - cursor)
            indices.extend(order[cursor:cursor + take].tolist())
            cursor += take
        for index in indices:
            exposures[index] += 1
        hosts_batch = torch.cat([hosts[index // len(fit_messages)] for index in indices]).to(device)
        messages = torch.stack([fit_messages[index % len(fit_messages)] for index in indices]).to(device)
        model.watermark_encoder.train()
        model.watermark_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        water = model.watermark_encoder(hosts_batch, messages)
        logits = model.watermark_decoder(water["carrier"])
        loss = F.binary_cross_entropy_with_logits(logits, messages)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite overfit loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["train"]["grad_clip"],
                                             error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                log_event(run_dir / "metrics.jsonl", {"kind": "overfit_train", "diagnostic_step": step,
                          "metrics_before_update": message_metrics(logits, messages),
                          "delta_w_rms_before_update": rms(water["residual"]),
                          "grad_l2_before_clip": float(norm)})
            evaluate(step)
    # Deliberately not format_version=1: train.py must not resume this diagnostic as a baseline run.
    torch.save({"diagnostic_format_version": 1, "kind": "watermark_only_bce_overfit",
                "model": model.state_dict(), "source_config": deepcopy(config),
                "diagnostic_steps": steps, "fit_messages": fit_messages,
                "heldout_messages": heldout_messages,
                "pair_exposure_counts": exposures}, run_dir / "diagnostic_weights.pt")
    return history


def random_message_fit(model, rgb, validation_messages, config, steps, batch_size,
                       eval_batch_size, log_every, device, run_dir, training_seed):
    """Train on a fresh message stream for one host and test a fixed unseen bank."""
    if rgb.ndim != 4 or rgb.shape[:2] != (1, 3):
        raise ValueError("random_message_fit requires exactly one B=1 RGB host")
    if (validation_messages.ndim != 2 or validation_messages.shape[1] != 64
            or len(validation_messages) < 2):
        raise ValueError("Validation messages must have shape N x 64 with N >= 2")
    if not bool(((validation_messages == 0) | (validation_messages == 1)).all()):
        raise ValueError("Validation messages must contain only 0 and 1")
    validation_keys_list = [tuple(int(value) for value in message.tolist())
                            for message in validation_messages]
    if len(set(validation_keys_list)) != len(validation_keys_list):
        raise ValueError("Validation messages must be unique")
    if min(steps, batch_size, eval_batch_size, log_every) < 1:
        raise ValueError("steps/batch-size/eval-batch-size/log-every must be >= 1")
    model.eval()
    for module in (model.color_encoder, model.color_decoder):
        module.requires_grad_(False)
    with torch.no_grad():
        host = model.color_encoder(*rgb_to_ycbcr(rgb.to(device)))["carrier"].detach()
    parameters = list(model.watermark_encoder.parameters()) + list(model.watermark_decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["train"]["lr"],
                                 weight_decay=config["train"]["weight_decay"])
    training_generator = torch.Generator().manual_seed(training_seed)
    validation_keys = set(validation_keys_list)
    seen_messages = set(validation_keys)
    history = []

    def evaluate(step):
        validation = evaluate_random_messages(
            model, host, validation_messages, device, eval_batch_size)
        event = {
            "kind": "random_message_validation",
            "diagnostic_step": step,
            "training_samples_seen": len(seen_messages) - len(validation_keys),
            "validation_messages_same_image": validation["matched"],
            "mismatched_labels_same_image": validation["mismatched_labels"],
            "watermark_removed_same_image": validation["watermark_removed"],
        }
        history.append(event)
        log_event(run_dir / "metrics.jsonl", event)

    evaluate(0)
    for step in range(1, steps + 1):
        messages = _draw_unique_messages(batch_size, training_generator, seen_messages).to(device)
        hosts = host.expand(batch_size, -1, -1, -1)
        model.watermark_encoder.train()
        model.watermark_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        water = model.watermark_encoder(hosts, messages)
        logits = model.watermark_decoder(water["carrier"])
        loss = F.binary_cross_entropy_with_logits(logits, messages)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite random-message loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["train"]["grad_clip"],
                                             error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                log_event(run_dir / "metrics.jsonl", {
                    "kind": "random_message_train",
                    "diagnostic_step": step,
                    "training_samples_seen": len(seen_messages) - len(validation_keys),
                    "batch_metrics_before_update": message_metrics(logits, messages),
                    "delta_w_rms_before_update": rms(water["residual"]),
                    "grad_l2_before_clip": float(norm),
                })
            evaluate(step)
    # Deliberately incompatible with the formal train.py checkpoint loader.
    torch.save({"diagnostic_format_version": 1,
                "kind": "watermark_only_bce_random_messages",
                "model": model.state_dict(), "source_config": deepcopy(config),
                "diagnostic_steps": steps, "training_seed": training_seed,
                "training_messages_seen": len(seen_messages) - len(validation_keys),
                "validation_messages": validation_messages},
               run_dir / "diagnostic_weights.pt")
    return history


def diagnostic_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect clean watermark sensitivity, overfit a small bank, or train on fresh random messages")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("inspect", "overfit", "random_messages"), default="inspect")
    parser.add_argument("--data-dir", required=True, help="Use training images, not the validation/test split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--messages", type=int,
                        help="Inspect/overfit only: distinct fixed messages per image; default 4")
    parser.add_argument("--validation-messages", type=int,
                        help="Overfit/random_messages: fixed unseen validation messages; "
                             "defaults to --messages for overfit and 256 for random_messages")
    parser.add_argument("--eval-batch-size", type=int,
                        help="Overfit/random_messages: validation batch size; default training batch size")
    parser.add_argument("--image-size", type=int, help="Default: checkpoint size; a change is diagnostic only")
    parser.add_argument("--steps", type=int, help="Required for training modes; forbidden for inspect")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", help="Must be a NEW directory; defaults to runs/watermark_MODE_timestamp")
    args = parser.parse_args(argv)
    if min(args.images, args.batch_size, args.log_every) < 1:
        parser.error("images/batch-size/log-every must be >= 1")
    if args.mode == "inspect" and args.steps is not None:
        parser.error("inspect does not train: omit --steps")
    if args.mode in ("overfit", "random_messages") and (args.steps is None or args.steps < 1):
        parser.error(f"{args.mode} requires explicit --steps >= 1")
    if args.mode in ("inspect", "overfit"):
        args.messages = 4 if args.messages is None else args.messages
        if args.messages < 2:
            parser.error("messages must be >= 2")
    if args.mode == "random_messages":
        if args.images != 1:
            parser.error("random_messages isolates one fixed host: pass --images 1")
        if args.messages is not None:
            parser.error("random_messages has no fixed fit bank: omit --messages")
        validation_count = 256 if args.validation_messages is None else args.validation_messages
        eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
        if validation_count < 2 or eval_batch_size < 1:
            parser.error("validation-messages must be >= 2 and eval-batch-size must be >= 1")
    elif args.mode == "overfit":
        validation_count = args.messages if args.validation_messages is None else args.validation_messages
        eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
        if validation_count < 2 or eval_batch_size < 1:
            parser.error("validation-messages must be >= 2 and eval-batch-size must be >= 1")
    else:
        if args.validation_messages is not None or args.eval_batch_size is not None:
            parser.error("validation-messages/eval-batch-size require a training diagnostic mode")
    checkpoint = load_checkpoint(args.checkpoint)
    config = deepcopy(checkpoint["config"])
    if config["channel"] != {"quantization_mode": "none", "clamp_enabled": False, "attack_mode": "identity"}:
        raise ValueError("These probes require a float + identity checkpoint without clamp")
    size = config["data"]["image_size"] if args.image_size is None else args.image_size
    dataset = ImageFolderDataset(args.data_dir, size, training=False, seed=config["seed"])
    if args.images > len(dataset):
        raise ValueError(f"Requested {args.images} images but directory contains {len(dataset)}")
    run_dir = Path(args.output_dir or (Path("runs") / (
        "watermark_" + args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))).resolve()
    if run_dir.exists():
        raise ValueError(f"Output directory already exists; choose a NEW directory: {run_dir}")
    seed_everything(config["seed"])
    device = resolve_device(args.device)
    model = DualPayloadSystem(config["model"], config["channel"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    rgbs = [dataset[index]["rgb"][None] for index in range(args.images)]
    fit = heldout = validation_messages = None
    fit_seed = validation_seed = training_seed = validation_message_policy = None
    if args.mode == "inspect":
        fit, heldout = message_banks(args.messages, config["seed"] + 700000)
    elif args.mode == "overfit":
        fit_seed = config["seed"] + 700000
        if args.validation_messages is None:
            validation_seed = fit_seed
            fit, heldout = message_banks(args.messages, fit_seed)
            validation_message_policy = "legacy_disjoint_suffix_from_fit_rng"
        else:
            validation_seed = config["seed"] + 800000
            fit, heldout = fixed_codebook_banks(
                args.messages, validation_count, fit_seed, validation_seed)
            validation_message_policy = "fixed_independent_bank_shared_with_random_messages"
    else:
        validation_seed = config["seed"] + 800000
        training_seed = config["seed"] + 900000
        validation_messages = random_message_bank(validation_count, validation_seed)
    metadata = {"mode": args.mode, "source_checkpoint": str(Path(args.checkpoint).resolve()),
                "source_global_step": checkpoint["global_step"], "source_config": config,
                "device": str(device), "torch": str(torch.__version__), "image_size": size,
                "images": [str(path) for path in dataset.paths[:args.images]],
                "crop": "fixed_center", "cli": vars(args),
                "objective": {"inspect": "no_updates", "overfit": "watermark_BCE_only",
                              "random_messages": "watermark_BCE_only_fresh_random_messages"}[args.mode],
                "optimizer": None if args.mode == "inspect" else "fresh_Adam_checkpoint_lr_and_weight_decay",
                "note": "Diagnostic only; message validation uses the same image and is not a cross-image test."}
    if args.mode == "inspect":
        metadata.update(fit_messages=fit.int().tolist(),
                        heldout_messages=heldout.int().tolist())
    elif args.mode == "overfit":
        metadata.update(
            fit_messages=fit.int().tolist(), heldout_messages=heldout.int().tolist(),
            fit_message_count=len(fit), validation_message_count=len(heldout),
            fit_seed=fit_seed, validation_seed=validation_seed,
            validation_message_policy=validation_message_policy,
            training_sampling_policy="balanced_shuffled_cartesian_cycles",
            eval_batch_size=eval_batch_size)
    else:
        metadata.update(
            host_policy="single_cached_color_carrier",
            training_message_policy="fresh_unique_random_each_step_excluding_validation",
            training_seed=training_seed, validation_seed=validation_seed,
            validation_message_count=validation_count,
            validation_messages=validation_messages.int().tolist(),
            eval_batch_size=eval_batch_size)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "manifest.json", metadata)
    log_event(run_dir / "metrics.jsonl", {"kind": "diagnostic_start", "mode": args.mode,
                                         "output_dir": str(run_dir)})
    if args.mode == "inspect":
        reports = []
        for index, rgb in enumerate(rgbs):
            report = {"kind": "message_sensitivity", "image_index": index,
                      **inspect_image(model, rgb.to(device), fit)}
            reports.append(report)
            log_event(run_dir / "metrics.jsonl", report)
        # Same small batch size as the baseline by default; gradients are not optimizer history.
        n = min(args.batch_size, len(rgbs))
        grad_messages = fit[torch.arange(n) % len(fit)].to(device)
        gradients = gradient_probe(model, torch.cat(rgbs[:n]).to(device), grad_messages, config)
        log_event(run_dir / "metrics.jsonl", {"kind": "gradient_probe", "batch_size": n, **gradients})
        write_json(run_dir / "report.json", {"metadata": metadata, "sensitivity": reports,
                                             "gradients": gradients})
    elif args.mode == "overfit":
        history = overfit(model, rgbs, fit, heldout, config, args.steps, args.batch_size,
                          args.log_every, device, run_dir, eval_batch_size)
        write_json(run_dir / "report.json", {"metadata": metadata, "evaluations": history})
    else:
        history = random_message_fit(model, rgbs[0], validation_messages, config, args.steps,
                                     args.batch_size, eval_batch_size, args.log_every, device,
                                     run_dir, training_seed)
        write_json(run_dir / "report.json", {"metadata": metadata, "evaluations": history})
    print(f"Completed. Diagnostic report: {run_dir / 'report.json'}", flush=True)
