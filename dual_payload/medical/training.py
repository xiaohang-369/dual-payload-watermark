"""FP32 medical training, explicit stages, validation, and resumable checkpoints."""
import copy
import json
import math
import platform
from pathlib import Path
from importlib.metadata import version

import torch
from torch.utils.data import DataLoader

from .artifacts import atomic_checkpoint, checked_checkpoint, code_digest, load_record, save_record, sha256
from .data import MedicalDataset
from .learning import (make_models, initialize_models, state_digest, ResidualQuantizer,
                       PayloadFactory, experiment_forward, validate_initialization)
from .profile import ARCHITECTURE, template, validate_config
from .optimization import options, validate_options, make_optimizer, schedule_step, augment_batch

STAGES = {'color': ('ec', 'dc'), 'transport': ('ew', 'dw'), 'decoder': ('dc',),
          'joint': ('ec', 'ew', 'dc', 'dw')}
LOSS_NAMES = {'color': {'rgb'}, 'transport': {'color_bits', 'patient_bits', 'gray', 'range'},
              'decoder': {'rgb'}, 'joint': {'rgb', 'color_bits', 'patient_bits', 'gray', 'range'}}


def training_template():
    return {'schema': 'medical-training-v1', 'purpose': 'experiment', 'stage': None,
            'device': 'cuda:0', 'threads': 1, 'seed': 2026,
            'image_size': [256, 256], 'precision': 'fp32', 'augmentation': 'none',
            'optimization': options(),
            'manifest': None, 'roots': {'pad': None}, 'calibration': None,
            'initialization': {'kind': 'scratch', 'path': None, 'sha256': None},
            'contract': {'profile_id': None, 'quantization_id': None,
                         'interleaver_seeds': {'color': 20260930, 'patient': 20260931}},
            'rms_limits': {'color_residual': None, 'color_ciphertext': None, 'patient_ciphertext': None},
            'epochs': None, 'max_steps': None, 'batch_size': None, 'micro_batch_size': None,
            'lr': None, 'weight_decay': 0.0, 'grad_clip': 1.0, 'save_every': None,
            'loss_weights': {}, 'output': None}


def validate_training(config, allow_test=False):
    if set(config) != set(training_template()) or config['schema'] != 'medical-training-v1':
        raise ValueError('Missing/unknown training configuration fields')
    if config['purpose'] not in ('experiment', 'test') or config['purpose'] == 'test' and not allow_test:
        raise ValueError('Test training requires explicit allow_test')
    if config['stage'] not in STAGES:
        raise ValueError('Choose color, transport, decoder, or joint stage explicitly')
    for key in ('threads', 'epochs', 'max_steps', 'batch_size', 'micro_batch_size', 'save_every'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Explicit positive integer required: {key}')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if config['micro_batch_size'] > config['batch_size']:
        raise ValueError('micro_batch_size exceeds logical batch_size')
    for key in ('lr', 'weight_decay', 'grad_clip'):
        v = config[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (key != 'weight_decay' and v == 0):
            raise ValueError(f'Explicit finite optimizer value required: {key}')
    weights = config['loss_weights']
    if set(weights) != LOSS_NAMES[config['stage']] or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError('Set the exact stage loss names and finite nonnegative weights')
    if config['stage'] in ('color', 'decoder', 'joint') and weights.get('rgb', 0) <= 0:
        raise ValueError('Color learning requires positive rgb loss')
    if config['stage'] in ('transport', 'joint') and any(weights.get(k, 0) <= 0 for k in ('color_bits', 'patient_bits')):
        raise ValueError('Both payload branches require positive loss weights')
    for key in ('manifest', 'calibration', 'output'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f'Explicit path required: {key}')
    if not isinstance(config['roots'], dict) or not config['roots'].get('pad'):
        raise ValueError('PAD root is required')
    if set(config['contract']) != {'profile_id', 'quantization_id', 'interleaver_seeds'}:
        raise ValueError('Unsupported transport contract')
    # Reuse wire validation without requiring final trained hashes to construct models.
    candidate = template()
    candidate.update(config['contract'], purpose=config['purpose'], rms_limits=config['rms_limits'],
                     quantization_steps=[1.0] * 39, weights={k: '0' * 64 for k in ('ec', 'ew', 'dc', 'dw')})
    validate_config(candidate, allow_test=allow_test)  # Validation sentinel only, never registered or saved.
    validate_initialization(config['initialization'])
    validate_options(config)


def configure_device(config):
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'])
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Medical experiments support cpu and cuda devices')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def _loader(dataset, config, epoch, training):
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                      num_workers=0, generator=generator)


def _microbatches(batch, size):
    n = len(batch['rgb'])
    for start in range(0, n, size):
        end = min(start + size, n)
        yield {key: value[start:end] for key, value in batch.items()}, (end-start)/n


def _event(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


@torch.no_grad()
def validate_epoch(models, quantizer, payloads, dataset, config):
    models.eval()
    total, count = {}, 0
    for batch in _loader(dataset, config, 0, False):
        for micro, _ in _microbatches(batch, config['micro_batch_size']):
            values = experiment_forward(models, quantizer, payloads, micro, config, validation=True)
            n = len(micro['rgb']); count += n
            for key, value in values.items():
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise RuntimeError('Nonfinite validation result')
                total[key] = total.get(key, 0.0) + n * scalar
    return {key: value/count for key, value in total.items()}


def train(config, *, allow_test=False, resume=None, resume_sha256=None, stop_after=None):
    """stop_after bounds a preflight run, while max_steps remains its resume target."""
    validate_training(config, allow_test)
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after must be a positive integer')
    if (resume is None) != (resume_sha256 is None):
        raise ValueError('Resume requires both checkpoint path and SHA-256')
    device = configure_device(config)
    training = MedicalDataset(config['manifest'], config['roots'], 'train')
    validation = MedicalDataset(config['manifest'], config['roots'], 'valid')
    calibration = load_record(config['calibration'])
    if calibration['schema'] != 'medical-calibration-v1' or calibration['split'] != 'train' or calibration['manifest_sha256'] != training.manifest['sha256']:
        raise ValueError('Calibration must belong to this manifest training split')
    if calibration['rms_limits']['color_residual'] != config['rms_limits']['color_residual']:
        raise ValueError('Ec RMS limit differs from calibration')
    if calibration.get('seed') != config['seed']:
        raise ValueError('Initialization seed differs from calibration')
    models = make_models(config['rms_limits'], seed=config['seed'])
    initialization = initialize_models(models, config['initialization'])
    if state_digest(models['ec']) != calibration['ec_state_sha256']:
        raise ValueError('Initial Ec differs from calibrated Ec; recalibrate this initialization')
    models.to(device)
    active = STAGES[config['stage']]
    for name, model in models.items():
        model.requires_grad_(name in active)
    parameters = [p for p in models.parameters() if p.requires_grad]
    optimizer = make_optimizer(models, config)
    total_updates = min(config['max_steps'], config['epochs'] * math.ceil(len(training)/config['batch_size']))
    quantizer = ResidualQuantizer(calibration['steps']).to(device)
    output = Path(config['output']).resolve()
    next_epoch, next_batch, step, best = 0, 0, 0, None
    if resume:
        state = checked_checkpoint(resume, resume_sha256)
        if state.get('schema') != 'medical-checkpoint-v1' or state['config'] != config:
            raise ValueError('Resume configuration must match the checkpoint exactly')
        if state['manifest_sha256'] != training.manifest['sha256'] or state['calibration_sha256'] != calibration['sha256']:
            raise ValueError('Resume input artifacts changed')
        if state['code_sha256'] != code_digest():
            raise ValueError('Code changed since checkpoint; use a new run with explicit medical initialization')
        if Path(resume).resolve().parent != output or not (output / 'run.json').is_file():
            raise ValueError('Resume must use its original run directory, including validation bank')
        models.load_state_dict(state['models'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, next_epoch, next_batch, best = state['step'], state['next_epoch'], state['next_batch'], state['best']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_record(output / 'run.json', {'config': config, 'manifest_sha256': training.manifest['sha256'],
                    'calibration_sha256': calibration['sha256'], 'initialization': initialization,
                    'code_sha256': code_digest(),
                    'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                'numpy': version('numpy'), 'pillow': version('pillow'),
                                'cryptography': version('cryptography'),
                                'sionna': version('sionna'), 'device': str(device),
                                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}})
    def checkpoint_state(metrics):
        return {'schema': 'medical-checkpoint-v1', 'architecture': ARCHITECTURE,
                'config': copy.deepcopy(config), 'models': models.state_dict(),
                'code_sha256': code_digest(), 'optimizer': optimizer.state_dict(),
                'step': step, 'best': best, 'next_epoch': next_epoch, 'next_batch': next_batch,
                'manifest_sha256': training.manifest['sha256'], 'data_manifest': training.manifest,
                'calibration_sha256': calibration['sha256'], 'steps': calibration['steps'],
                'validation': metrics, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}

    # Recover even when interrupted before the first validation/save interval.
    if not resume:
        atomic_checkpoint(output / 'last.pt', checkpoint_state({}))
    payloads = PayloadFactory(config['contract'], output / 'nonces.sqlite', str(device))
    limit = min(config['max_steps'], step + stop_after) if stop_after is not None else config['max_steps']
    if step >= limit or next_epoch >= config['epochs']:
        raise ValueError('No remaining training steps/epochs')
    for epoch in range(next_epoch, config['epochs']):
        loader = _loader(training, config, epoch, True)
        for batch_index, batch in enumerate(loader):
            if epoch == next_epoch and batch_index < next_batch:
                continue
            for name, model in models.items():
                model.train(name in active)
            schedule_step(optimizer, config, step, total_updates)
            if config['augmentation'] == 'dihedral':
                batch = augment_batch(batch, config['seed'], epoch)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            logged = {}
            for micro, scale in _microbatches(batch, config['micro_batch_size']):
                values = experiment_forward(models, quantizer, payloads, micro, config, epoch=epoch)
                if not torch.isfinite(values['total']):
                    raise RuntimeError('Nonfinite training loss')
                (values['total'] * scale).backward()
                for key, value in values.items():
                    logged[key] = logged.get(key, 0.0) + float(value.detach()) * scale
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            rates = {g['group_name']: g['lr'] for g in optimizer.param_groups}
            memory = ({'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                       'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
                      if device.type == 'cuda' else {})
            _event(output / 'metrics.jsonl', {'kind': 'train', 'step': step, 'epoch': epoch,
                                             'gradient_norm': float(norm), 'learning_rates': rates,
                                             **memory, **logged})
            at_end = batch_index + 1 == len(loader)
            if step % config['save_every'] == 0 or at_end or step >= limit:
                metrics = validate_epoch(models, quantizer, payloads, validation, config)
                _event(output / 'metrics.jsonl', {'kind': 'validation', 'step': step, **metrics})
                improved = best is None or metrics['total'] < best
                best = metrics['total'] if improved else best
                state = checkpoint_state(metrics)
                state.update(next_epoch=epoch + 1 if at_end else epoch,
                             next_batch=0 if at_end else batch_index + 1)
                if improved:
                    atomic_checkpoint(output / 'best.pt', state)
                atomic_checkpoint(output / 'last.pt', state)
            if step >= limit:
                return {'step': step, 'checkpoint': str(output / 'last.pt'),
                        'sha256': sha256(output / 'last.pt'), 'validation': metrics}
        next_batch = 0
    return {'step': step, 'checkpoint': str(output / 'last.pt'), 'sha256': sha256(output / 'last.pt')}
