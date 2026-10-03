"""Joint training or frozen Protocol evaluation; evaluation has no optimizer."""
import torch
from .config import validate_config

NETWORKS = ("color_encoder", "watermark_encoder", "color_decoder", "watermark_decoder")


def apply_training_stage(model, config: dict) -> None:
    validate_config(config)
    if model.message_bits != 256:
        raise ValueError("Medical V3 requires 256-bit models")
    stage = config["train"]["stage"]
    model.train(stage != "protocol_eval")
    for name in NETWORKS:
        module = getattr(model, name)
        enabled = stage == "joint_256"
        module.requires_grad_(enabled)
        module.train(enabled)
        for parameter in module.parameters():
            parameter.grad = None


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(model, config):
    apply_training_stage(model, config)
    if config["train"]["stage"] == "protocol_eval":
        raise ValueError("protocol_eval is evaluation only; no optimizer")
    return torch.optim.Adam(trainable_parameters(model), lr=config["train"]["lr"],
                            weight_decay=config["train"]["weight_decay"])
