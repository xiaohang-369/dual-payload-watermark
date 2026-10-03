"""Strict medical V3 configuration; omitted payload/stage fields never migrate."""
from copy import deepcopy
import json
import math
from pathlib import Path

STAGES = ("joint_256", "protocol_eval")
DEFAULT_CONFIG = {
    "seed": 2026, "device": "auto",
    "model": {"architecture_version": "v3", "message_bits": 256,
              "delta_c": 2 / 255, "delta_w": 2 / 255, "eps": 1e-8},
    "channel": {"quantization_mode": "none", "clamp_enabled": False, "attack_mode": "identity"},
    "loss": {"rgb": 1., "chroma": 0., "luma": 0., "message": 1., "carrier": 1., "range": .1},
    "data": {"train_manifest": None, "val_manifest": None, "image_size": 256,
             "batch_size": 2, "num_workers": 0},
    "train": {"stage": "joint_256", "key_transform": False, "epochs": 20,
              "lr": 1e-4, "weight_decay": 0., "grad_clip": 1., "max_steps": None,
              "log_every": 10, "output_dir": None},
    "protocol": {"candidate_count": 15, "beta_c": None, "beta_m": None,
                 "min_moved_c": None, "min_moved_m": None},
}


def message_bits(model_config: dict) -> int:
    value = model_config.get("message_bits")
    if type(value) is not int or value != 256:
        raise ValueError("model.message_bits must be explicitly set to integer 256")
    return value


def architecture_version(config: dict) -> str:
    return config["model"]["architecture_version"]


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate_protocol(config: dict, *, required: bool = True):
    if type(config.get("candidate_count")) is not int or config["candidate_count"] != 15:
        raise ValueError("Protocol v1 candidate_count must be 15")
    for suffix, maximum in (("c", 39), ("m", 9)):
        beta, moved = config.get(f"beta_{suffix}"), config.get(f"min_moved_{suffix}")
        if beta is not None or required:
            finite_nonnegative(beta, f"beta_{suffix}")
        if moved is not None or required:
            if type(moved) is not int or not 1 <= moved <= maximum:
                raise ValueError(f"min_moved_{suffix} must be an integer in 1..{maximum}")


def validate_config(config: dict) -> None:
    message_bits(config.get("model", {}))
    if config["model"].get("architecture_version") != "v3":
        raise ValueError("model.architecture_version must be v3 (v2clean backbone)")
    stage = config.get("train", {}).get("stage")
    if stage not in STAGES:
        raise ValueError(f"train.stage must be explicit: {STAGES}")
    if config["train"].get("key_transform") is not (stage == "protocol_eval"):
        raise ValueError("key_transform must be true only in protocol_eval")
    if config.get("channel") != DEFAULT_CONFIG["channel"]:
        raise ValueError("Medical V3 requires FP32 identity channel without clamp or quantization")
    if config["data"]["image_size"] != 256:
        raise ValueError("Prepared medical work images must be 256x256")
    for key in ("delta_c", "delta_w", "eps"):
        finite_nonnegative(config["model"][key], key)
    if config["model"]["eps"] == 0:
        raise ValueError("eps must be positive")
    for key, value in config["loss"].items():
        finite_nonnegative(value, f"loss.{key}")
    if stage == "joint_256":
        for key in ("rgb", "message", "carrier", "range"):
            value = config["loss"].get(key)
            finite_nonnegative(value, f"loss.{key}")
            if value == 0:
                raise ValueError(f"joint_256 requires loss.{key} to be finite and strictly positive")
    for group, keys in (("train", ("epochs", "log_every")), ("data", ("batch_size",))):
        for key in keys:
            if type(config[group][key]) is not int or config[group][key] < 1:
                raise ValueError(f"{group}.{key} must be a positive integer")
    if type(config["data"]["num_workers"]) is not int or config["data"]["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("lr", "weight_decay", "grad_clip"):
        finite_nonnegative(config["train"][key], key)
    if config["train"]["lr"] == 0:
        raise ValueError("lr must be positive")
    maximum = config["train"]["max_steps"]
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("max_steps must be positive or null")
    validate_protocol(config["protocol"], required=stage == "protocol_eval")


def _merge(base, update, prefix=""):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown config field: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be an object")
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def load_config(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    message_bits(raw.get("model", {}))
    if raw.get("train", {}).get("stage") not in STAGES:
        raise ValueError("train.stage must be explicitly provided")
    if not isinstance(raw.get("protocol"), dict) or not DEFAULT_CONFIG["protocol"].keys() <= raw["protocol"].keys():
        raise ValueError("Explicit protocol fields are required; use null for pending training-stage parameters")
    config = deepcopy(DEFAULT_CONFIG)
    _merge(config, raw)
    validate_config(config)
    return config
