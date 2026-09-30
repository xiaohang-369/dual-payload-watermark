"""Train-only frequency quantiles; refuse degenerate encoders and missing steps."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..transforms import BlockDCT, rgb_to_ycbcr
from .artifacts import save_record
from .learning import make_models, initialize_models, state_digest
from .protocol import COLOR_INDICES


@torch.no_grad()
def calibrate(dataset, initialization, limits, output, *, quantile, minimum_rms, device='cpu'):
    if dataset.split != 'train':
        raise ValueError('Quantization calibration is restricted to the training split')
    if not 0 < quantile < 1 or not np.isfinite(minimum_rms) or minimum_rms <= 0:
        raise ValueError('Explicit quantile in (0,1) and positive minimum_rms required')
    models = make_models(limits)
    report = initialize_models(models, initialization)
    ec = models['ec'].to(device).eval()
    dct = BlockDCT().to(device)
    rms_sum, count = 0.0, 0
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='medical-calibration-', dir=Path(output).parent) as work:
        coefficients = np.memmap(Path(work) / 'coefficients.bin', mode='w+', dtype=np.float32,
                                 shape=(len(dataset) * 1024, 39))
        for index in range(len(dataset)):
            rgb = dataset[index]['rgb'].unsqueeze(0).to(device)
            residual = ec(*rgb_to_ycbcr(rgb))['residual']
            rms_sum += float(residual.square().mean()); count += 1
            selected = dct(residual)[0, list(COLOR_INDICES)].permute(1, 2, 0).reshape(1024, 39)
            coefficients[index * 1024:(index+1) * 1024] = selected.cpu().numpy()
        rms = (rms_sum / count) ** 0.5
        if not np.isfinite(coefficients).all() or rms <= minimum_rms:
            raise ValueError('Ec residual is nonfinite or degenerate; verify pretrained weights before calibration')
        coverage = np.array([np.quantile(np.abs(coefficients[:, i]), quantile) for i in range(39)])
        steps = coverage / 7.0  # Conservative positive endpoint; negative endpoint remains -8.
        if not np.isfinite(steps).all() or np.any(steps <= 0):
            raise ValueError('At least one frequency has no usable range; calibration not frozen')
        clipped, squared_error = 0, 0.0
        for start in range(0, len(coefficients), 8192):
            a = np.asarray(coefficients[start:start+8192])
            rounded = np.rint(a / steps)
            q = np.clip(rounded, -8, 7)
            clipped += int(((rounded < -8) | (rounded > 7)).sum())
            squared_error += float(np.square(a - q * steps).sum())
        size = int(coefficients.size)
        del coefficients
    return save_record(output, {'schema': 'medical-calibration-v1', 'split': 'train',
                               'manifest_sha256': dataset.manifest['sha256'],
                               'ec_state_sha256': state_digest(ec), 'initialization': report,
                               'rms_limits': limits, 'quantile': quantile, 'minimum_rms': minimum_rms,
                               'images': count, 'residual_rms': rms, 'steps': steps.tolist(),
                               'clipped_fraction': clipped / size, 'coefficient_mse': squared_error / size})
