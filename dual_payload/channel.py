"""Separate storage precision and attacks. Only identity attacks ship in V1."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class AttackResult:
    attacked_image: Tensor
    targets: dict[str, Tensor]
    valid_mask: Tensor
    attack_info: dict


class IdentityAttack(nn.Module):
    def forward(self, image: Tensor, targets: dict[str, Tensor]) -> AttackResult:
        return AttackResult(image, targets, torch.ones_like(image), {"type": "identity"})


class TransmissionChannel(nn.Module):
    def __init__(self, quantization_mode: str = "none", clamp_enabled: bool = False,
                 attack_mode: str = "identity") -> None:
        super().__init__()
        if quantization_mode not in ("none", "ste8", "real8"):
            raise ValueError("quantization_mode must be none, ste8 or real8")
        if quantization_mode != "none" and not clamp_enabled:
            raise ValueError("8-bit modes require clamp_enabled=true")
        if attack_mode != "identity":
            raise NotImplementedError("Only identity is implemented; no silent attack fallback")
        self.quantization_mode, self.clamp_enabled = quantization_mode, clamp_enabled
        self.attack = IdentityAttack()

    def quantize(self, x: Tensor) -> Tensor:
        value = x.clamp(0, 1) if self.clamp_enabled else x
        if self.quantization_mode == "none":
            return value
        rounded = (value * 255).round() / 255
        if self.quantization_mode == "ste8":
            # Identity backward through rounding; the clamp keeps its true gradient.
            return value + (rounded - value).detach()
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("real8 is evaluation-only; use ste8 for training")
        return rounded

    def forward(self, x: Tensor, targets: dict[str, Tensor]) -> tuple[Tensor, AttackResult]:
        x_quantized = self.quantize(x)
        result = self.attack(x_quantized, targets)
        result.attack_info.update(quantization_mode=self.quantization_mode,
                                  clamp_enabled=self.clamp_enabled)
        return x_quantized, result
