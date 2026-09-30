"""Differentiable color codec and detached, real encrypted transport samples."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..transforms import BlockDCT, rgb_to_ycbcr
from ..system import DualPayloadSystem
from .artifacts import checked_checkpoint
from .crypto import NonceStore, encrypt_frame
from .ldpc import TransportCodec
from .models import ColorEncoder, ColorDecoder, WatermarkEncoder, WatermarkDecoder, import_v2_color_weights
from .preprocess import ste_gray8
from .profile import ARCHITECTURE
from .protocol import Branch, BLOCKS, COLOR_INDICES, pack_color


@dataclass
class TrainingLayout:
    """Only transport permutations; no frozen weights or signed profile identity."""
    permutations: tuple
    inverses: tuple

    def permutation(self, branch, inverse=False):
        return (self.inverses if inverse else self.permutations)[int(branch) - 1]


def training_layout(seeds):
    permutations = tuple(np.random.Generator(np.random.PCG64(seeds[name])).permutation(BLOCKS[b] * 1536)
                         for name, b in (('color', Branch.COLOR), ('patient', Branch.PATIENT)))
    return TrainingLayout(permutations, tuple(np.argsort(p) for p in permutations))


def make_models(limits):
    if not isinstance(limits, dict) or set(limits) != {
        'color_residual', 'color_ciphertext', 'patient_ciphertext'
    } or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1
             for v in limits.values()):
        raise ValueError('All three finite RMS limits must be explicitly set in (0,1]')
    return nn.ModuleDict({'ec': ColorEncoder(limits['color_residual']),
                          'ew': WatermarkEncoder(limits['color_ciphertext'], limits['patient_ciphertext']),
                          'dc': ColorDecoder(), 'dw': WatermarkDecoder()})


def initialize_models(models, source):
    if source['kind'] == 'scratch':
        if source.get('path') is not None or source.get('sha256') is not None:
            raise ValueError('Scratch initialization must not specify checkpoint material')
        return {'kind': 'scratch', 'loaded': [], 'initialized': list(models.state_dict())}
    checkpoint = checked_checkpoint(source['path'], source['sha256'])
    if source['kind'] == 'v2':
        if checkpoint.get('architecture_version') != 'v2':
            raise ValueError('Expected a verified V2 checkpoint')
        original = DualPayloadSystem(checkpoint['config']['model'])
        original.load_state_dict(checkpoint['model'], strict=True)
        report = import_v2_color_weights(models['ec'], models['dc'], original.state_dict())
        report['initialized'].extend('ew.' + k for k in models['ew'].state_dict())
        report['initialized'].extend('dw.' + k for k in models['dw'].state_dict())
        return dict(report, kind='v2', source_sha256=source['sha256'])
    if source['kind'] != 'medical' or checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Unsupported initialization checkpoint')
    models.load_state_dict(checkpoint['models'], strict=True)
    return {'kind': 'medical', 'source_sha256': source['sha256'], 'loaded': list(models.state_dict())}


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class ResidualQuantizer(nn.Module):
    def __init__(self, steps):
        super().__init__()
        values = torch.tensor(steps, dtype=torch.float32)
        if values.shape != (39,) or not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError('39 finite positive steps required')
        self.register_buffer('steps', values.reshape(1, 39, 1, 1))
        self.dct = BlockDCT()

    def forward(self, residual, ste=True):
        selected = self.dct(residual)[:, list(COLOR_INDICES)]
        rounded = torch.round(selected / self.steps)
        integers = rounded.clamp(-8, 7)
        exact = integers * self.steps
        selected_hat = selected + (exact - selected).detach() if ste else exact
        full = residual.new_zeros(residual.shape[0], 64, 32, 32)
        full[:, list(COLOR_INDICES)] = selected_hat
        return {'residual': self.dct.inverse(full), 'integers': integers.detach(),
                'clipped_fraction': ((rounded < -8) | (rounded > 7)).float().mean(),
                'coefficient_mse': (selected - exact).square().mean()}


class PayloadFactory:
    """Actual AES-GCM/LDPC samples; persist a fixed synthetic validation bank."""
    def __init__(self, contract, nonce_path, device):
        from pathlib import Path
        self.contract, self.device = contract, device
        self.codec = TransportCodec(training_layout(contract['interleaver_seeds']), device=device)
        self.nonces = NonceStore(nonce_path)
        self.bank = Path(nonce_path).parent / 'validation_frames'
        self.bank.mkdir(exist_ok=True)

    @torch.no_grad()
    def make(self, integers, sample_ids, seed, epoch=0, validation=False):
        import secrets
        from .artifacts import load_record, save_record
        from .profile import canonical_json
        colors, patients = [], []
        for q, sample in zip(integers, sample_ids):
            q = q.permute(1, 2, 0).cpu().numpy().astype(np.int8)
            color = pack_color(q, self.contract['quantization_id'])
            identity = canonical_json({'seed': seed, 'sample': sample, 'contract': self.contract}) + color
            bank_file = self.bank / (hashlib.sha256(identity).hexdigest() + '.json')
            if validation and bank_file.exists():
                record = load_record(bank_file)
                frames = [bytes.fromhex(record[name]) for name in ('color', 'patient')]
            else:
                token, image_id = secrets.token_bytes(16), secrets.token_bytes(16)
                frames = [encrypt_frame(plaintext, secrets.token_bytes(32), branch,
                                        self.contract['profile_id'], image_id, self.nonces)
                          for branch, plaintext in ((Branch.COLOR, color), (Branch.PATIENT, token))]
                if validation:
                    save_record(bank_file, dict(color=frames[0].hex(), patient=frames[1].hex()))
            colors.append(self.codec.encode(frames[0], Branch.COLOR))
            patients.append(self.codec.encode(frames[1], Branch.PATIENT))
        return torch.cat(colors), torch.cat(patients)


def masked_mean(error, rectangles):
    values = []
    for item, rect in zip(error, rectangles):
        x, y, w, h = map(int, rect)
        values.append(item[:, y:y+h, x:x+w].mean())
    return torch.stack(values).mean()


def experiment_forward(models, quantizer, payloads, batch, config, *, validation=False, epoch=0):
    rgb = batch['rgb'].to(config['device'])
    y, cb, cr = rgb_to_ycbcr(rgb)
    residual = models['ec'](y, cb, cr)['residual']
    quantized = quantizer(residual, ste=not validation)
    losses = {}
    if config['stage'] == 'color':
        gray = ste_gray8(y)
    else:
        c, m = payloads.make(quantized['integers'], batch['image_id'], config['seed'], epoch, validation)
        carrier = models['ew'](y, c, m)['carrier']
        gray = ste_gray8(carrier)
        logits = models['dw'](gray)
        for name, truth, branch in (('color_bits', c, Branch.COLOR), ('patient_bits', m, Branch.PATIENT)):
            prediction = logits['color_logits' if branch == Branch.COLOR else 'patient_logits'].flatten(1)[:, :-512]
            target = truth.flatten(1)[:, :-512]
            losses[name] = F.binary_cross_entropy_with_logits(prediction, target)
            losses[name + '_ber'] = ((prediction >= 0) != target.bool()).float().mean()
        losses['gray'] = masked_mean((gray-y).square(), batch['content_rect'])
        losses['range'] = (F.relu(-carrier).square() + F.relu(carrier-1).square()).mean()
    if config['stage'] != 'transport':
        # This is a supervised color-codec path, not claimed as extraction through AES.
        # Freeze carrier condition for decoder adaptation; Ec gradients go via STE residual.
        estimate = models['dc'](gray.detach(), quantized['residual'])['rgb']
        losses['rgb'] = masked_mean((estimate-rgb).abs(), batch['content_rect'])
    total = sum(weight * losses[name] for name, weight in config['loss_weights'].items() if weight)
    losses.update(total=total, quantization_clipped=quantized['clipped_fraction'],
                  quantization_mse=quantized['coefficient_mse'])
    return losses
