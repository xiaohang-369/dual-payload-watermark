"""Shared full-reference image metrics; callers select whole images or content regions."""

import torch
from torch import Tensor
from torch.nn import functional as F


def psnr_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    mse = (prediction - target).square().flatten(1).mean(1)
    # Finite JSON/logs: the reporting ceiling is 120 dB, including exact matches.
    return -10 * mse.clamp_min(1e-12).log10()


def psnr(prediction: Tensor, target: Tensor) -> Tensor:
    return psnr_per_sample(prediction, target).mean()


def ssim_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    """Single-scale SSIM: data_range=1, Gaussian 11/1.5, valid windows, channel mean."""
    size = min(11, *prediction.shape[-2:])
    size -= 1 - size % 2
    coords = torch.arange(size, dtype=prediction.dtype, device=prediction.device) - size // 2
    gaussian = torch.exp(-coords.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    channels = prediction.shape[1]
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(channels, 1, size, size)

    def blur(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, groups=channels)

    mx, my = blur(prediction), blur(target)
    vx = (blur(prediction.square()) - mx.square()).clamp_min(0)
    vy = (blur(target.square()) - my.square()).clamp_min(0)
    covariance = blur(prediction * target) - mx * my
    score = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2))
    return score.flatten(1).mean(1)


def ssim(prediction: Tensor, target: Tensor) -> Tensor:
    return ssim_per_sample(prediction, target).mean()
