"""Connect both decoders to the exact same channel output."""

import torch
from torch import Tensor, nn

from .channel import TransmissionChannel
from .models import ColorDecoder, ColorEncoder, WatermarkDecoder, WatermarkEncoder
from .transforms import rgb_to_ycbcr


class DualPayloadSystem(nn.Module):
    def __init__(self, model_config: dict | None = None, channel_config: dict | None = None) -> None:
        super().__init__()
        config = ({"architecture_version": "v2"}
                  if model_config is None else model_config)
        architecture_version = config.get("architecture_version", "v1")
        if architecture_version != "v2":
            raise ValueError(
                "This implementation constructs Network V2 only; set "
                "model.architecture_version=v2 explicitly"
            )
        if any(key in config for key in ("channels", "blocks")):
            raise ValueError("model.channels and model.blocks are V1-only architecture fields")
        self.architecture_version = architecture_version
        budgets = {key: config[key] for key in ("eps",) if key in config}
        self.color_encoder = ColorEncoder(**budgets, delta_c=config.get("delta_c", 2 / 255))
        self.watermark_encoder = WatermarkEncoder(**budgets, delta_w=config.get("delta_w", 2 / 255))
        self.color_decoder = ColorDecoder()
        self.watermark_decoder = WatermarkDecoder()
        self.channel = TransmissionChannel(**(channel_config or {}))

    def decode(self, attacked_image: Tensor) -> dict[str, Tensor]:
        """Blind inference: neither target, message nor valid mask is an input."""
        color = self.color_decoder(attacked_image)
        return {"rgb_hat": color["rgb"], "y_hat": color["y"],
                "cb_hat": color["cb"], "cr_hat": color["cr"],
                "logits": self.watermark_decoder(attacked_image)}

    def forward(self, rgb: Tensor, message: Tensor) -> dict:
        if rgb.dtype != torch.float32:
            raise ValueError("Network V2 runs in FP32; pass float32 RGB and do not enable AMP")
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 256, 256):
            raise ValueError("Network V2 requires RGB with shape B x 3 x 256 x 256")
        if not bool(torch.isfinite(rgb).all()) or not bool(((rgb >= 0) & (rgb <= 1)).all()):
            raise ValueError("Input RGB must be finite normalized sRGB in [0, 1]")
        y, cb, cr = rgb_to_ycbcr(rgb)
        color = self.color_encoder(y, cb, cr)
        water = self.watermark_encoder(color["carrier"], message)
        targets = {"rgb": rgb, "luma": y, "chroma": torch.cat((cb, cr), dim=1)}
        xq, channel = self.channel(water["carrier"], targets)
        return {
            **self.decode(channel.attacked_image),
            "y": y, "cb": cb, "cr": cr, "s": color["carrier"],
            "x_float": water["carrier"], "x_quantized": xq,
            "attacked_image": channel.attacked_image,
            "delta_c": color["residual"], "delta_w": water["residual"],
            "target_rgb": channel.targets["rgb"], "target_luma": channel.targets["luma"],
            "target_chroma": channel.targets["chroma"], "valid_mask": channel.valid_mask,
            "attack_info": channel.attack_info,
        }
