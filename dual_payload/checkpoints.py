"""Strict medical V3 checkpoints; no partial loads or legacy migration."""
import hashlib
import io
from pathlib import Path
import torch
from .config import validate_config
from .system import DualPayloadSystem
from .transforms import BlockDCT

SCHEMA = "medical-v3-256-v1"


def validate_model_buffers(model):
    reference = BlockDCT().state_dict()
    for module in model.modules():
        if isinstance(module, BlockDCT):
            for name, value in module.state_dict().items():
                if value.dtype != reference[name].dtype or not torch.equal(value.cpu(), reference[name]):
                    raise ValueError(f"Incompatible Protocol v1 DCT buffer: {name}")
    if any(parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all())
           for parameter in model.parameters()):
        raise ValueError("Model parameters must be finite FP32")


def load_checkpoint_bytes(raw):
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise ValueError("Expected a medical V3 checkpoint; legacy checkpoints are not supported")
    validate_config(checkpoint["config"])
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint must contain all four networks and buffers")
    return checkpoint


def load_checkpoint(path):
    return load_checkpoint_bytes(Path(path).read_bytes())


def load_weights(model, checkpoint):
    validate_config(checkpoint["config"])
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("Expected a medical V3 checkpoint")
    if model.model_config != checkpoint["config"]["model"]:
        raise ValueError("Checkpoint model configuration must match exactly")
    target, source = model.state_dict(), checkpoint["model"]
    if target.keys() != source.keys():
        raise ValueError("Checkpoint state keys must match exactly")
    for name, value in source.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[name].shape or value.dtype != target[name].dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
    model.load_state_dict(source, strict=True)
    validate_model_buffers(model)


def model_from_file(path, device="cpu"):
    # Digest and load the same byte snapshot, avoiding two separate file reads.
    raw = Path(path).read_bytes()
    checkpoint = load_checkpoint_bytes(raw)
    config = checkpoint["config"]
    model = DualPayloadSystem(config["model"], config["channel"])
    load_weights(model, checkpoint)
    return model.to(device), config, hashlib.sha256(raw).digest()


def save_checkpoint(path, model, config, **training_state):
    validate_config(config)
    validate_model_buffers(model)
    if model.model_config != config["model"]:
        raise ValueError("Saved configuration must match the model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**training_state, "schema": SCHEMA, "config": config, "model": model.state_dict()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
