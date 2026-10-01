"""Explicit optimization and geometry for the single-device main experiment."""
import hashlib
import math

import torch


def options():
    return {'name': 'adam', 'betas': [0.9, 0.999],
            'schedule': 'constant', 'warmup_steps': 0, 'minimum_lr': 1e-6}


def validate_options(config):
    settings = config['optimization']
    if not isinstance(settings, dict) or set(settings) != set(options()):
        raise ValueError('Explicit optimization fields required')
    if settings['name'] not in ('adam', 'adamw') or settings['schedule'] not in ('constant', 'warmup_cosine'):
        raise ValueError('Unsupported optimizer or schedule')
    if (not isinstance(settings['betas'], list) or len(settings['betas']) != 2 or
            any(type(v) not in (float, int) or not 0 <= v < 1 for v in settings['betas'])):
        raise ValueError('Two optimizer betas in [0,1) required')
    if type(settings['warmup_steps']) is not int or settings['warmup_steps'] < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    for key in ('minimum_lr',):
        value = settings[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite learning rate required: ' + key)
    if settings['minimum_lr'] > config['lr']:
        raise ValueError('minimum_lr exceeds a group learning rate')
    if config['image_size'] != [256, 256] or config['precision'] != 'fp32':
        raise ValueError('Medical V1 requires 256x256 FP32 working images')
    if config['augmentation'] not in ('none', 'dihedral'):
        raise ValueError('Unsupported geometric augmentation')


def make_optimizer(models, config):
    settings = config['optimization']
    parameters = [{'params': [p for p in models.parameters() if p.requires_grad],
                   'lr': config['lr'], 'initial_lr': config['lr'], 'group_name': 'all'}]
    cls = torch.optim.AdamW if settings['name'] == 'adamw' else torch.optim.Adam
    return cls(parameters, betas=tuple(settings['betas']), weight_decay=config['weight_decay'])


def schedule_step(optimizer, config, completed, total):
    """Derive LR from the absolute update number, including after resume."""
    settings = config['optimization']
    if settings['schedule'] == 'constant':
        return
    if not 0 <= completed < total:
        raise ValueError('Schedule step outside training horizon')
    warmup = min(settings['warmup_steps'], max(0, total - 1))
    for group in optimizer.param_groups:
        base = group['initial_lr']
        if completed < warmup:
            group['lr'] = base * (0.1 + 0.9 * (completed + 1) / warmup)
        else:
            progress = (completed - warmup) / max(1, total - warmup - 1)
            group['lr'] = settings['minimum_lr'] + (base-settings['minimum_lr']) * (1+math.cos(math.pi*progress))/2


def transform_geometry(rgb, rect, code):
    """Exact pixel permutations; carry the valid-content rectangle with the image."""
    x, y, w, h = map(int, rect)
    height, width = rgb.shape[-2:]
    if code & 1:
        rgb = rgb.flip(-1); x = width - x - w
    if code & 2:
        rgb = rgb.flip(-2); y = height - y - h
    for _ in range((code >> 2) % 4):
        rgb = torch.rot90(rgb, 1, (-2, -1))
        x, y, w, h = y, width - x - w, h, w
        height, width = width, height
    return rgb, (x, y, w, h)


def augment_batch(batch, seed, epoch):
    result = dict(batch)
    images, rectangles = [], []
    for image, rect, identity in zip(batch['rgb'], batch['content_rect'], batch['image_id']):
        code = hashlib.sha256(f'{seed}:{epoch}:{identity}'.encode()).digest()[0] % 16
        image, rect = transform_geometry(image, rect, code)
        images.append(image); rectangles.append(rect)
    result['rgb'] = torch.stack(images)
    result['content_rect'] = torch.tensor(rectangles, dtype=batch['content_rect'].dtype)
    return result
