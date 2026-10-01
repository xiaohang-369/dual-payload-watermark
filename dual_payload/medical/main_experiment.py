"""PAD main experiment on one H100: one joint run of all four networks."""
import argparse
import fcntl
import json
from pathlib import Path

import torch

from .artifacts import checked_checkpoint, code_digest, load_record, save_record, sha256
from .calibration import calibrate
from .data import MedicalDataset, create_manifest, SOURCES
from .profile import read_json
from .training import train, training_template, configure_device, validate_training


def resolve_config(pad_root, output, *, config_path,
                   profile_id=None, quantization_id=None, micro_batch_size=None):
    """Read hyperparameters from JSON, then bind runtime paths and explicit overrides."""
    config = read_json(config_path)
    if not isinstance(config, dict) or set(config) != set(training_template()):
        raise ValueError('Missing/unknown training configuration fields')
    root = Path(output).resolve()
    config.update(manifest=str(root / 'manifest.json'), roots={'pad': str(Path(pad_root).resolve())},
                  calibration=str(root / 'calibration.json'), output=str(root / 'joint'))
    if micro_batch_size is not None:
        config['micro_batch_size'] = micro_batch_size
    contract = config['contract']
    if not isinstance(contract, dict):
        raise ValueError('Unsupported transport contract')
    for key, override in (('profile_id', profile_id), ('quantization_id', quantization_id)):
        if override is not None:
            contract[key] = override
        elif contract.get(key) is None:
            contract[key] = 1
    validate_training(config)
    if config['stage'] != 'joint':
        raise ValueError('The main experiment requires stage=joint')
    if config['initialization'] != {'kind': 'scratch', 'path': None, 'sha256': None}:
        raise ValueError('The main experiment requires scratch initialization with null path/sha256')
    if config['augmentation'] != 'none':
        raise ValueError('The main experiment requires augmentation=none')
    return config


def write_config(path, config):
    if path.exists():
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
             config_path, profile_id=None, quantization_id=None, micro_batch_size=None):
    config = resolve_config(pad_root, output, config_path=config_path,
                            profile_id=profile_id, quantization_id=quantization_id,
                            micro_batch_size=micro_batch_size)
    device = torch.device(config['device'])
    if device.type != 'cuda':
        raise ValueError('The main experiment requires a CUDA H100 device')
    if not torch.cuda.is_available():
        raise ValueError('The main experiment requires the H100 server; CUDA is unavailable here')
    properties = torch.cuda.get_device_properties(device)
    if 'H100' not in properties.name or properties.total_memory < 70 * 1024**3:
        raise ValueError('A full H100 80GB GPU is required; check device allocation/MIG')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.main.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another process is already running this experiment') from exc
        return _run_locked(pad_root, metadata, root, config)


def _run_locked(pad_root, metadata, root, config):
    pad_root, metadata = map(lambda p: str(Path(p).resolve()), (pad_root, metadata))
    identity = {'training_mode': 'joint', 'pad_root': pad_root, 'metadata': metadata, 'metadata_sha256': sha256(metadata),
                'initialization': 'scratch', 'code_sha256': code_digest(),
                'profile_id': config['contract']['profile_id'],
                'quantization_id': config['contract']['quantization_id'],
                'micro_batch_size': config['micro_batch_size'],
                'fractions': [0.7, 0.15, 0.15], 'seed': config['seed'], 'config': config}
    record = root / 'main-inputs.json'
    if record.exists():
        saved = load_record(record); saved.pop('sha256')
        if saved != identity:
            raise ValueError('Main experiment inputs changed; use a new output directory')
    else:
        save_record(record, identity)
    write_config(root / 'train-joint.json', config)
    manifest = root / 'manifest.json'
    if not manifest.exists():
        created = create_manifest(pad_root, metadata, manifest, (0.7, 0.15, 0.15), config['seed'])
        print(json.dumps({'event': 'manifest', 'counts': created['counts'],
                          'diagnoses': created['diagnoses']}), flush=True)
    document = load_record(manifest)
    if (document['sources']['pad'] != {'url': SOURCES['pad'], 'metadata_sha256': identity['metadata_sha256']} or
            document['seed'] != config['seed'] or document['pad_group_fractions'] != [0.7, 0.15, 0.15] or
            any(r['dataset'] != 'pad' for r in document['records'])):
        raise ValueError('Main manifest differs from the planned PAD-only training split')
    source = config['initialization']
    calibration_path = root / 'calibration.json'
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
    parser.add_argument('--config', required=True, help='Training JSON: the source of training hyperparameters')
    parser.add_argument('--profile-id', type=int, help='Override JSON profile ID (null defaults to 1)')
    parser.add_argument('--quantization-id', type=int, help='Override JSON quantization ID (null defaults to 1)')
    parser.add_argument('--micro-batch-size', type=int, help='Override JSON micro-batch size')
    args = parser.parse_args(argv)
    result = run_main(args.pad_root, args.pad_metadata, args.output,
                      config_path=args.config,
                      profile_id=args.profile_id, quantization_id=args.quantization_id,
                      micro_batch_size=args.micro_batch_size)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
