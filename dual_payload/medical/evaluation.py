"""Freeze public artifacts and evaluate actual PNGs with a separate receiver process."""
import json
import math
import os
import secrets
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PIL import Image

from ..metrics import psnr_per_sample, ssim_per_sample
from ..transforms import rgb_to_ycbcr
from .artifacts import checked_checkpoint, save_record, load_record, sha256
from .crypto import NonceStore
from .data import MedicalDataset
from .learning import make_models
from .models import load_component
from .pipeline import Sender
from .preprocess import integer_pixels
from .profile import ARCHITECTURE, template, register_profile, load_profile
from .protocol import Branch, BLOCKS, bytes_to_bits


def evaluation_policy_template():
    return {'schema': 'medical-evaluation-policy-v1',
            'thresholds': {'color_packet_success_min': None, 'patient_packet_success_min': None,
                           'color_recovery_success_min': None,
                           'joint_success_min': None, 'gray_content_psnr_min': None,
                           'rgb_content_psnr_min': None},
            'rgb_quality': 'conditional-on-exact-color-packet-success',
            'thresholds_per_modality': True,
            'psnr_ceiling_db': 120, 'color_difference': 'CIE76-on-clipped-sRGB-D65',
            'regions': ['whole', 'content'], 'modalities': ['clinical', 'dermoscopic']}


def validate_policy(policy):
    fixed = evaluation_policy_template()
    if set(policy) != set(fixed) or any(policy[k] != fixed[k] for k in fixed if k != 'thresholds'):
        raise ValueError('Unsupported evaluation policy')
    if set(policy['thresholds']) != set(fixed['thresholds']):
        raise ValueError('Set all evaluation thresholds before formal testing')
    for key, value in policy['thresholds'].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or ('success' in key and value > 1):
            raise ValueError(f'Explicit finite evaluation threshold required: {key}')


def export_experiment(checkpoint_path, expected_sha256, output, policy, *, allow_test=False, profile_id=None):
    validate_policy(policy)
    if profile_id is not None and (type(profile_id) is not int or not 1 <= profile_id <= 65535):
        raise ValueError('Published profile_id must be an integer in [1,65535]')
    state = checked_checkpoint(checkpoint_path, expected_sha256)
    if state.get('schema') != 'medical-checkpoint-v1' or state.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected medical training checkpoint')
    config = state['config']
    if config['purpose'] == 'test' and not allow_test:
        raise ValueError('Test checkpoints require allow_test')
    models = make_models(config['rms_limits'])
    models.load_state_dict(state['models'], strict=True)
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    weights = {}
    for name, model in models.items():
        path = destination / (name + '.pt')
        torch.save({'architecture': ARCHITECTURE, 'component': name, 'state_dict': model.state_dict()}, path)
        weights[name] = sha256(path)
    public = template()
    public.update(config['contract'], purpose=config['purpose'], quantization_steps=state['steps'],
                  rms_limits=config['rms_limits'], weights=weights)
    if profile_id is not None:
        public['profile_id'] = profile_id
    registered = register_profile(public, destination / 'profiles', allow_test=allow_test)
    profile = load_profile(registered, allow_test=allow_test)
    save_record(destination / 'data_manifest.json', state['data_manifest'])
    save_record(destination / 'evaluation_policy.json', policy)
    return save_record(destination / 'export.json', {'schema': 'medical-export-v1',
                       'checkpoint_sha256': expected_sha256, 'training_step': state['step'],
                       'training_stage': config['stage'],
                       'training_code_sha256': state['code_sha256'],
                       'manifest_sha256': state['manifest_sha256'],
                       'profile_path': registered.relative_to(destination).as_posix(),
                       'profile_sha256': profile.digest.hex(), 'weights': weights,
                       'policy_sha256': load_record(destination / 'evaluation_policy.json')['sha256'],
                       'validation': state['validation']})


def _lab(rgb):
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    matrix = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / rgb.new_tensor([0.95047, 1., 1.08883]).reshape(1, 3, 1, 1)
    delta = 6 / 29
    f = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1/3), xyz / (3*delta**2) + 4/29)
    return torch.stack((116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])), dim=1)


def image_metrics(prediction, target, rect):
    result = {}
    x, y, w, h = map(int, rect)
    for region, pred, ref in (('whole', prediction, target),
                              ('content', prediction[:, :, y:y+h, x:x+w], target[:, :, y:y+h, x:x+w])):
        error = (pred-ref).abs()
        values = {'mse': float(error.square().mean()), 'mae': float(error.mean()),
                  'psnr': float(psnr_per_sample(pred, ref).mean()),
                  'ssim': float(ssim_per_sample(pred, ref).mean()), 'max_error': float(error.max()),
                  'p99_error': float(torch.quantile(error.flatten(), 0.99)),
                  'out_of_range_fraction': float(((pred < 0) | (pred > 1)).float().mean())}
        if pred.shape[1] == 3:
            values['delta_e76'] = float((_lab(pred)-_lab(ref)).square().sum(1).sqrt().mean())
            py, pcb, pcr = rgb_to_ycbcr(pred); ry, rcb, rcr = rgb_to_ycbcr(ref)
            values['luma_mae'] = float((py-ry).abs().mean())
            values['chroma_mae'] = float(torch.cat((pcb-rcb, pcr-rcr), 1).abs().mean())
        result[region] = values
    return result


def transport_metrics(reference, diagnostics):
    result = {}
    for name, branch in (('color', Branch.COLOR), ('patient', Branch.PATIENT)):
        expected = reference[name + '_bits'].cpu().numpy().reshape(-1)[:-512]
        expected_info = bytes_to_bits(reference[name + '_frame'] + bytes(BLOCKS[branch] * 128 - len(reference[name + '_frame']))).reshape(-1, 1024)
        if name + '_logits' not in diagnostics:
            result[name] = {'ber': None, 'block_failures': None, 'block_count': BLOCKS[branch]}
            continue
        logits = diagnostics[name + '_logits'].reshape(-1)[:-512]
        values = {'ber': float(((logits >= 0) != expected).mean()), 'block_count': BLOCKS[branch],
                  'block_failures': None, 'packet_exact': False}
        if name + '_information' in diagnostics:
            actual = diagnostics[name + '_information']
            if actual.shape != expected_info.shape:
                raise ValueError('Invalid receiver diagnostic shape')
            failures = int(np.any(actual != expected_info, axis=1).sum())
            values.update(block_failures=failures, packet_exact=failures == 0)
        result[name] = values
    return result


def summarize(rows, thresholds):
    n = len(rows)
    if not n:
        raise ValueError('Cannot evaluate an empty cohort')
    result = {'samples': n, 'authentication_success': sum(r['authentication'] == 'AUTHENTIC' for r in rows)/n,
              'color_packet_success': sum(r['color_ok'] for r in rows)/n,
              'patient_packet_success': sum(r['patient_ok'] for r in rows)/n,
              'joint_success': sum(r['color_ok'] and r['patient_ok'] for r in rows)/n}
    result['color_aead_success'] = sum(r.get('color_aead', False) for r in rows)/n
    result['patient_aead_success'] = sum(r.get('patient_aead', False) for r in rows)/n
    result['color_recovery_success'] = sum('rgb_float' in r for r in rows)/n
    result['evaluation_errors'] = sum(r.get('evaluation_error', False) for r in rows)
    for branch in ('color', 'patient'):
        measured = [r['transport'][branch] for r in rows if r.get('transport', {}).get(branch, {}).get('ber') is not None]
        result[branch + '_ber'] = sum(r['ber'] for r in measured)/len(measured) if measured else None
        result[branch + '_ber_samples'] = len(measured)
        decoded = [r for r in measured if r['block_failures'] is not None]
        result[branch + '_block_failure_rate'] = (sum(r['block_failures'] for r in decoded)/sum(r['block_count'] for r in decoded)) if decoded else None
        result[branch + '_decoded_samples'] = len(decoded)
    for name in ('gray', 'rgb_float', 'rgb_integer'):
        available = [r[name] for r in rows if name in r]
        result[name + '_samples'] = len(available)
        result[name] = {region: {key: sum(r[region][key] for r in available)/len(available)
                                  for key in available[0][region]} for region in ('whole', 'content')} if available else None
    result['checks'] = {name: result[name] >= thresholds[name + '_min']
                        for name in ('color_packet_success', 'patient_packet_success', 'joint_success', 'color_recovery_success')}
    result['checks']['gray_content_psnr'] = (result['gray'] is not None and result['gray']['content']['psnr'] >= thresholds['gray_content_psnr_min'])
    result['checks']['rgb_content_psnr'] = (result['rgb_integer'] is not None and result['rgb_integer']['content']['psnr'] >= thresholds['rgb_content_psnr_min'])
    result['passed'] = all(result['checks'].values()) and result['authentication_success'] == 1 and result['evaluation_errors'] == 0
    result['failures'] = [{'image_id': r['image_id'], 'reason': r.get('reason'),
                           'color': r.get('color_status'), 'patient': r.get('patient_status'),
                           'color_reason': r.get('color_reason'), 'patient_reason': r.get('patient_reason')}
                          for r in rows if not (r['color_ok'] and r['patient_ok'] and 'rgb_float' in r) or r.get('evaluation_error')]
    result['worst_gray'] = [{'image_id': r['image_id'], 'content_psnr': r['gray']['content']['psnr']}
                            for r in sorted((r for r in rows if 'gray' in r),
                                            key=lambda r: r['gray']['content']['psnr'])[:10]]
    result['worst_rgb'] = [{'image_id': r['image_id'], 'content_psnr': r['rgb_integer']['content']['psnr']}
                           for r in sorted((r for r in rows if 'rgb_integer' in r),
                                           key=lambda r: r['rgb_integer']['content']['psnr'])[:10]]
    return result


def evaluate_export(export_dir, manifest, roots, split, output, *, device='cpu', allow_test=False, receiver_timeout=300):
    if split not in ('valid', 'test', 'external'):
        raise ValueError('File evaluation accepts valid, test, or external splits')
    export_dir = Path(export_dir).resolve()
    exported = load_record(export_dir / 'export.json')
    profile_dir = (export_dir / exported['profile_path']).resolve()
    if not profile_dir.is_relative_to(export_dir):
        raise ValueError('Unsafe exported profile path')
    profile = load_profile(profile_dir, allow_test=allow_test)
    if profile.digest.hex() != exported['profile_sha256']:
        raise ValueError('Exported profile changed')
    policy = load_record(export_dir / 'evaluation_policy.json')
    if policy.pop('sha256') != exported['policy_sha256']:
        raise ValueError('Evaluation policy changed after model export')
    validate_policy(policy)
    dataset = MedicalDataset(manifest, roots, split)
    baseline_manifest = load_record(export_dir / 'data_manifest.json')
    if baseline_manifest['sha256'] != exported['manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    if split != 'external' and dataset.manifest['sha256'] != exported['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from the frozen training manifest')
    if split == 'external':
        original_pad = [r for r in baseline_manifest['records'] if r['dataset'] == 'pad']
        current_pad = [r for r in dataset.manifest['records'] if r['dataset'] == 'pad']
        if original_pad != current_pad:
            raise ValueError('Adding external data must preserve the original PAD records and split')
        seen = {r[key] for r in original_pad for key in ('file_sha256', 'work_sha256')}
        if any(r[key] in seen for r in dataset.records for key in ('file_sha256', 'work_sha256')):
            raise ValueError('External images overlap the main experiment')
    destination = Path(output).resolve(); destination.mkdir(parents=True, exist_ok=False)
    private = destination / 'private'; private.mkdir(mode=0o700)
    ckey, mkey, signing = secrets.token_bytes(32), secrets.token_bytes(32), Ed25519PrivateKey.generate()
    for name, value in (('color.key', ckey), ('patient.key', mkey)):
        (private / name).write_bytes(value); (private / name).chmod(0o600)
    public_path = destination / 'hospital.pem'
    public_path.write_bytes(signing.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    sender = Sender(profile, load_component('ec', export_dir / 'ec.pt', profile, device),
                    load_component('ew', export_dir / 'ew.pt', profile, device),
                    NonceStore(private / 'nonces.sqlite'), device)
    save_record(destination / 'evaluation_run.json', {'export_sha256': exported['sha256'],
                'manifest_sha256': dataset.manifest['sha256'], 'split': split, 'policy': policy,
                'device': device, 'receiver': 'independent-process'})
    rows = []
    for index in range(len(dataset)):
        source = dataset.records[index]
        row = {'image_id': source['dataset'] + ':' + source['image_id'], 'modality': source['modality'],
               'authentication': 'NOT_RUN', 'color_ok': False, 'patient_ok': False}
        try:
            sample = dataset[index]
            rgb = sample['rgb'].unsqueeze(0); rect = sample['content_rect'].tolist()
            token, reference = secrets.token_bytes(16), {}
            shared = destination / f'{index:06d}.png'
            row['png'] = shared.name
            sent = sender.send(rgb, token, ckey, mkey, signing, shared, tuple(rect),
                               observer=lambda event, values: reference.update(values))
            row['send'] = sent
            np.savez_compressed(private / f'{index:06d}-reference.npz',
                                color_bits=reference['color_bits'].cpu().numpy(),
                                patient_bits=reference['patient_bits'].cpu().numpy(),
                                color_frame=np.frombuffer(reference['color_frame'], dtype=np.uint8),
                                patient_frame=np.frombuffer(reference['patient_frame'], dtype=np.uint8))
            with Image.open(shared) as image:
                gray = torch.tensor(np.array(image), dtype=torch.float32)[None, None]/255
            row['gray'] = image_metrics(gray, rgb_to_ycbcr(rgb)[0], rect)
            received = destination / f'{index:06d}-received'
            cmd = [sys.executable, '-m', 'dual_payload.medical.cli', 'receive', '--profile', str(profile_dir),
                   '--input', str(shared), '--output', str(received), '--hospital-public-key', str(public_path),
                   '--color-key', str(private / 'color.key'), '--patient-key', str(private / 'patient.key'),
                   '--dw', str(export_dir / 'dw.pt'), '--dc', str(export_dir / 'dc.pt'), '--device', device,
                   '--diagnostics']
            if allow_test: cmd.append('--allow-test-profile')
            # Supply only the public Python package location, not source images or sender state.
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
            completed = subprocess.run(cmd, cwd=export_dir, env=environment,
                                       capture_output=True, text=True, timeout=receiver_timeout)
            if completed.returncode not in (0, 2) or not (received / 'status.json').is_file():
                raise RuntimeError('Receiver process error: ' + completed.stderr[-2000:])
            status = json.loads((received / 'status.json').read_text())
            row.update(authentication=status['authentication'], color_status=status['color']['status'],
                       patient_status=status['patient']['status'], reason=status['reason'],
                       color_reason=status['color']['reason'], patient_reason=status['patient']['reason'])
            with np.load(received / 'diagnostics.npz', allow_pickle=False) as diagnostics:
                row['transport'] = transport_metrics(reference, diagnostics)
                row['color_aead'] = bool(diagnostics.get('color_aead_verified', False))
                row['patient_aead'] = bool(diagnostics.get('patient_aead_verified', False))
            authenticated = status['authentication'] == 'AUTHENTIC'
            row['color_ok'] = authenticated and row['color_aead'] and row['transport']['color'].get('packet_exact', False)
            row['patient_ok'] = authenticated and status['patient']['status'] == 'OK' and (received / 'patient.token').read_bytes() == token
            if row['color_ok'] and status['color']['status'] == 'OK':
                restored = torch.from_numpy(np.load(received / 'rgb_float.npy', allow_pickle=False))[None]
                row['rgb_float'] = image_metrics(restored, rgb, rect)
                integer = torch.tensor(integer_pixels(restored), dtype=torch.float32)/255
                row['rgb_integer'] = image_metrics(integer, rgb, rect)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            row['reason'] = str(exc)
            row['evaluation_error'] = True
        rows.append(row)
        with (destination / 'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    cohorts = defaultdict(list)
    for row in rows: cohorts[row['modality']].append(row)
    summary = {'split': split, 'overall': summarize(rows, policy['thresholds']),
               'modalities': {name: summarize(items, policy['thresholds']) for name, items in cohorts.items()}}
    summary['passed'] = summary['overall']['passed'] and all(v['passed'] for v in summary['modalities'].values())
    save_record(destination / 'summary.json', summary)
    return summary
