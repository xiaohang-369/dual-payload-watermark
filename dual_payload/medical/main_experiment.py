"""PAD main experiment on one H100: one joint run of all four networks."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from .artifacts import checked_checkpoint, code_digest, load_record, save_record, sha256
from .calibration import calibrate
from .data import MedicalDataset, create_manifest, SOURCES
from .training import train, training_template, configure_device, validate_training


def h100_config():
    config = training_template()
    config.update(stage='joint', epochs=120, max_steps=20000, lr=1e-4,
                  batch_size=16, micro_batch_size=4, weight_decay=1e-4, save_every=500,
                  augmentation='none', loss_weights={'rgb': 1.0, 'color_bits': 1.0,
                      'patient_bits': 1.0, 'gray': 1000.0, 'range': 1000.0},
                  rms_limits={'color_residual': 2/255, 'color_ciphertext': 2/255,
                              'patient_ciphertext': 1/255})
    config['optimization'].update(name='adamw', schedule='warmup_cosine', warmup_steps=200)
    return config


def write_config(path, config):
    if path.exists():
        from .profile import read_json
        if read_json(path) != config:
            raise ValueError('Existing main-run configuration differs: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write('\n')


def run_training(config):
    """Train once or resume this same joint run."""
    directory = Path(config['output'])
    receipt = directory.parent / (directory.name + '-complete.json')
    def check_inputs(state):
        if (state['manifest_sha256'] != load_record(config['manifest'])['sha256'] or
                state['calibration_sha256'] != load_record(config['calibration'])['sha256']):
            raise ValueError('Training manifest or calibration changed')
    if receipt.exists():
        result = load_record(receipt)
        state = checked_checkpoint(result['checkpoint'], result['checkpoint_sha256'])
        if state['config'] != config or state['code_sha256'] != code_digest():
            raise ValueError('Completed training configuration or code changed')
        check_inputs(state)
        return result
    last = directory / 'last.pt'
    if not last.exists():
        train(config)
    digest = sha256(last)
    state = checked_checkpoint(last, digest)
    if state['config'] != config or state['code_sha256'] != code_digest():
        raise ValueError('Existing training checkpoint configuration or code changed')
    check_inputs(state)
    if state['step'] < config['max_steps'] and state['next_epoch'] < config['epochs']:
        train(config, resume=str(last), resume_sha256=digest)
    best = directory / 'best.pt'
    final = checked_checkpoint(last, sha256(last))
    return save_record(receipt, {'checkpoint': str(best), 'checkpoint_sha256': sha256(best),
                                 'last_step': final['step'], 'stage': config['stage']})


def run_main(pad_root, metadata, output, *,
             profile_id=1, quantization_id=1, micro_batch_size=4):
    if micro_batch_size not in (1, 2, 4, 8, 16):
        raise ValueError('micro_batch_size must divide the effective batch of 16')
    if not torch.cuda.is_available():
        raise ValueError('The main experiment requires the H100 server; CUDA is unavailable here')
    properties = torch.cuda.get_device_properties(0)
    if 'H100' not in properties.name or properties.total_memory < 70 * 1024**3:
        raise ValueError('A full H100 80GB GPU is required; check device allocation/MIG')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.main.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is already running this experiment') from exc
        return _run_locked(pad_root, metadata, root,
                           profile_id, quantization_id, micro_batch_size)


def _run_locked(pad_root, metadata, root, profile_id, quantization_id, micro_batch_size):
    pad_root, metadata = map(lambda p: str(Path(p).resolve()), (pad_root, metadata))
    identity = {'training_mode': 'joint', 'pad_root': pad_root, 'metadata': metadata, 'metadata_sha256': sha256(metadata),
                'initialization': 'scratch', 'code_sha256': code_digest(),
                'profile_id': profile_id, 'quantization_id': quantization_id,
                'micro_batch_size': micro_batch_size, 'fractions': [0.7, 0.15, 0.15], 'seed': 2026}
    record = root / 'main-inputs.json'
    if record.exists():
        saved = load_record(record); saved.pop('sha256')
        if saved != identity:
            raise ValueError('Main experiment inputs changed; use a new output directory')
    else:
        save_record(record, identity)
    manifest = root / 'manifest.json'
    if not manifest.exists():
        created = create_manifest(pad_root, metadata, manifest, (0.7, 0.15, 0.15), 2026)
        print(json.dumps({'event': 'manifest', 'counts': created['counts'],
                          'diagnoses': created['diagnoses']}), flush=True)
    document = load_record(manifest)
    if (document['sources']['pad'] != {'url': SOURCES['pad'], 'metadata_sha256': identity['metadata_sha256']} or
            document['seed'] != 2026 or document['pad_group_fractions'] != [0.7, 0.15, 0.15] or
            any(r['dataset'] != 'pad' for r in document['records'])):
        raise ValueError('Main manifest differs from the planned PAD-only training split')
    source = {'kind': 'scratch', 'path': None, 'sha256': None}
    calibration_path = root / 'calibration.json'
    config = h100_config()
    config.update(manifest=str(manifest), roots={'pad': pad_root}, initialization=source,
                  output=str(root / 'joint'), micro_batch_size=micro_batch_size,
                  calibration=str(calibration_path))
    config['contract'].update(profile_id=profile_id, quantization_id=quantization_id)
    validate_training(config)
    write_config(root / 'train-joint.json', config)
    configure_device(config)
    # Steps are calibrated once from the initial Ec and remain fixed throughout training.
    if not calibration_path.exists():
        print(json.dumps({'event': 'calibration'}), flush=True)
        calibrate(MedicalDataset(manifest, config['roots'], 'train'), source,
                  config['rms_limits'], calibration_path, quantile=0.999,
                  minimum_rms=1e-6, device=config['device'], seed=config['seed'])
    calibration = load_record(calibration_path)
    if (calibration['manifest_sha256'] != document['sha256'] or
            calibration.get('seed') != config['seed'] or
            calibration['initialization']['kind'] != 'scratch' or
            calibration['rms_limits'] != config['rms_limits']):
        raise ValueError('Existing calibration differs from the scratch main experiment')
    print(json.dumps({'event': 'training', 'stage': 'joint', 'output': config['output']}), flush=True)
    result = run_training(config)
    return {'status': 'training_complete', 'training': result,
            'note': 'Final file evaluation and medical acceptance remain separate.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the PAD main experiment on H100 80GB')
    parser.add_argument('--pad-root', required=True)
    parser.add_argument('--pad-metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile-id', type=int, default=1)
    parser.add_argument('--quantization-id', type=int, default=1)
    parser.add_argument('--micro-batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    result = run_main(args.pad_root, args.pad_metadata, args.output,
                      profile_id=args.profile_id, quantization_id=args.quantization_id,
                      micro_batch_size=args.micro_batch_size)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
