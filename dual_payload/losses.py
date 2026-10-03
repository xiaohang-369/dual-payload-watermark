"""Clean training objectives; scalar weights are experimental configuration."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CleanLoss(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, output: dict, message: Tensor) -> dict[str, Tensor]:
        if output["attack_info"]["type"] != "identity" or not bool((output["valid_mask"] == 1).all()):
            raise NotImplementedError("CleanLoss needs identity/full-valid targets; define attack supervision first")
        x = output["x_float"]
        terms = {
            "rgb": F.l1_loss(output["rgb_hat"], output["target_rgb"]),
            "chroma": F.l1_loss(torch.cat((output["cb_hat"], output["cr_hat"]), 1), output["target_chroma"]),
            "luma": F.l1_loss(output["y_hat"], output["target_luma"]),
            "message": F.binary_cross_entropy_with_logits(output["logits"], message.float()),
            "carrier": F.mse_loss(x, output["y"]),
            "range": (F.relu(-x).square() + F.relu(x - 1).square()).mean(),
        }
        terms["total"] = sum(self.weights.get(key, 0.0) * value for key, value in terms.items()
                             if self.weights.get(key, 0.0) != 0)
        return terms
