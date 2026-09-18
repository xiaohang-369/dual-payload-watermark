"""Small, strict JSON configuration without a YAML dependency."""

from copy import deepcopy
import json
import math
from pathlib import Path


DEFAULT_CONFIG = {
    "seed": 2026,
    "device": "auto",
    "model": {"channels": 64, "blocks": 8, "delta_c": 2 / 255, "delta_w": 2 / 255, "eps": 1e-8},
    "channel": {"quantization_mode": "none", "clamp_enabled": False, "attack_mode": "identity"},
    "loss": {"rgb": 1.0, "chroma": 0.0, "luma": 0.0, "message": 1.0, "carrier": 1.0, "range": 0.1},
    "data": {"train_dir": None, "val_dir": None, "image_size": 256, "batch_size": 2, "num_workers": 0},
    "train": {"epochs": 20, "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 1.0,
              "max_steps": None, "log_every": 10, "output_dir": None},
    "experiment": None,
}


def _merge(base: dict, update: dict, prefix: str = "") -> None:
    if not isinstance(update, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    for key, value in update.items():
        name = prefix + key
        if key not in base:
            raise ValueError(f"Unknown config key: {name}")
        if isinstance(base[key], dict):
            _merge(base[key], value, name + ".")
        else:
            base[key] = value


def validate_config(config: dict) -> None:
    def positive_int(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def number(value, name, minimum=0.0, strict=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric")
        if value < minimum or (strict and value == minimum):
            raise ValueError(f"Invalid {name}: {value}")

    positive_int(config["seed"], "seed", 0)
    for key in ("channels", "blocks"):
        positive_int(config["model"][key], "model." + key)
    for key in ("delta_c", "delta_w", "eps"):
        number(config["model"][key], "model." + key, strict=(key == "eps"))
    for key, value in config["loss"].items():
        number(value, "loss." + key)
    if not any(config["loss"].values()):
        raise ValueError("At least one loss weight must be positive")
    data, training, channel = config["data"], config["train"], config["channel"]
    positive_int(data["image_size"], "data.image_size", 8)
    if data["image_size"] % 8:
        raise ValueError("data.image_size must be divisible by 8")
    positive_int(data["batch_size"], "data.batch_size")
    positive_int(data["num_workers"], "data.num_workers", 0)
    for key in ("train_dir", "val_dir"):
        if data[key] is not None and not isinstance(data[key], str):
            raise ValueError(f"data.{key} must be a path string or null")
    for key in ("epochs", "log_every"):
        positive_int(training[key], "train." + key)
    if training["max_steps"] is not None:
        positive_int(training["max_steps"], "train.max_steps")
    number(training["lr"], "train.lr", strict=True)
    number(training["weight_decay"], "train.weight_decay")
    number(training["grad_clip"], "train.grad_clip", strict=True)
    if training["output_dir"] is not None and not isinstance(training["output_dir"], str):
        raise ValueError("train.output_dir must be a path string or null")
    if channel["quantization_mode"] not in ("none", "ste8", "real8"):
        raise ValueError("Unknown quantization_mode")
    if not isinstance(channel["clamp_enabled"], bool):
        raise ValueError("clamp_enabled must be boolean")
    if channel["quantization_mode"] != "none" and not channel["clamp_enabled"]:
        raise ValueError("8-bit modes require clamp_enabled=true")
    if channel["attack_mode"] != "identity":
        raise ValueError("V1 implements identity attacks only")
    if not isinstance(config["device"], str) or not (
        config["device"] in ("auto", "cpu", "cuda") or config["device"].startswith("cuda:")):
        raise ValueError("device must be auto, cpu, cuda or cuda:N")
    experiment = config.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, dict):
            raise ValueError("experiment must be a JSON object or null")
        required = {
            "mode", "asset_manifest", "image_manifest", "message_bank", "message_metadata",
            "image_count", "message_count", "message_bits", "pair_count", "crop_policy",
        }
        missing, unknown = required - experiment.keys(), experiment.keys() - required
        if missing:
            raise ValueError("Missing experiment keys: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError("Unknown experiment keys: " + ", ".join(sorted(unknown)))
        for key in ("mode", "asset_manifest", "image_manifest", "message_bank",
                    "message_metadata", "crop_policy"):
            if not isinstance(experiment[key], str) or not experiment[key]:
                raise ValueError(f"experiment.{key} must be a non-empty string")
        for key in ("image_count", "message_count", "message_bits", "pair_count"):
            positive_int(experiment[key], "experiment." + key)
        if experiment["message_bits"] != 64:
            raise ValueError("experiment.message_bits must be 64")
        if experiment["pair_count"] != experiment["image_count"] * experiment["message_count"]:
            raise ValueError("experiment.pair_count must equal image_count * message_count")
        if experiment["crop_policy"] != "fixed_center_crop_256":
            raise ValueError("joint_10x20 requires crop_policy=fixed_center_crop_256")
        if config["data"]["image_size"] != 256:
            raise ValueError("joint_10x20 requires data.image_size=256")


def load_config(path: str | Path | None = None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open(encoding="utf-8-sig") as stream:
            _merge(config, json.load(stream))
    validate_config(config)
    return config
