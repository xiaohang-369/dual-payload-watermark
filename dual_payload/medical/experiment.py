"""Server-facing preparation, calibration, training, freeze and file evaluation."""
import argparse
import json
import torch

from .artifacts import save_record
from .calibration import calibrate
from .data import create_manifest, MedicalDataset
from .evaluation import export_experiment, evaluate_export
from .learning import make_models, initialize_models, state_digest
from .profile import read_json
from .training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description='Medical main experiment tools (no automatic downloads)')
    sub = parser.add_subparsers(dest='command', required=True)
    manifest = sub.add_parser('manifest')
    manifest.add_argument('--pad-root', required=True)
    manifest.add_argument('--pad-metadata', required=True)
    manifest.add_argument('--derm-root')
    manifest.add_argument('--derm-metadata')
    manifest.add_argument('--fractions', type=float, nargs=3, required=True, metavar=('TRAIN', 'VALID', 'TEST'))
    manifest.add_argument('--seed', type=int, required=True)
    manifest.add_argument('--output', required=True)
    audit = sub.add_parser('audit-weights')
    audit.add_argument('--config', required=True)
    audit.add_argument('--output', required=True)
    calibration = sub.add_parser('calibrate')
    calibration.add_argument('--config', required=True)
    calibration.add_argument('--quantile', type=float, required=True)
    calibration.add_argument('--minimum-rms', type=float, required=True)
    calibration.add_argument('--output', required=True)
    trainer = sub.add_parser('train')
    trainer.add_argument('--config', required=True)
    trainer.add_argument('--resume')
    trainer.add_argument('--resume-sha256')
    trainer.add_argument('--stop-after', type=int, help='Bound a preflight run without changing max_steps')
    trainer.add_argument('--allow-test', action='store_true')
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--sha256', required=True)
    export.add_argument('--policy', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--profile-id', type=int, help='Fresh publication ID; defaults to the training contract ID')
    export.add_argument('--allow-test', action='store_true')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--export', required=True)
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--pad-root', required=True)
    evaluate.add_argument('--derm-root')
    evaluate.add_argument('--split', choices=('valid', 'test', 'external'), required=True)
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--threads', type=int, default=1)
    evaluate.add_argument('--allow-test', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'manifest':
        result = create_manifest(args.pad_root, args.pad_metadata, args.output, args.fractions, args.seed,
                                 args.derm_root, args.derm_metadata)
        print(json.dumps({'sha256': result['sha256'], 'counts': result['counts'], 'missing_views': result['missing_views']}))
        return 0
    if args.command in ('calibrate', 'audit-weights', 'train'):
        config = read_json(args.config)
        if args.command == 'train':
            if args.stop_after is not None and args.stop_after <= 0:
                parser.error('--stop-after must be positive')
            result = train(config, allow_test=args.allow_test, resume=args.resume,
                           resume_sha256=args.resume_sha256, stop_after=args.stop_after)
        else:
            torch.set_num_threads(config['threads']); torch.manual_seed(config['seed'])
            if args.command == 'audit-weights':
                models = make_models(config['rms_limits'], seed=config['seed'])
                report = initialize_models(models, config['initialization'])
                result = save_record(args.output, {'schema': 'medical-weight-audit-v1', 'report': report,
                                     'state_sha256': {k: state_digest(v) for k, v in models.items()}})
            else:
                dataset = MedicalDataset(config['manifest'], config['roots'], 'train')
                result = calibrate(dataset, config['initialization'], config['rms_limits'], args.output,
                                   quantile=args.quantile, minimum_rms=args.minimum_rms,
                                   device=config['device'], seed=config['seed'])
    elif args.command == 'export':
        result = export_experiment(args.checkpoint, args.sha256, args.output,
                                   read_json(args.policy), allow_test=args.allow_test, profile_id=args.profile_id)
    else:
        if args.threads < 1:
            parser.error('--threads must be positive')
        torch.set_num_threads(args.threads)
        roots = {'pad': args.pad_root}
        if args.derm_root: roots['derm7pt'] = args.derm_root
        result = evaluate_export(args.export, args.manifest, roots, args.split, args.output,
                                 device=args.device, allow_test=args.allow_test)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.command == 'evaluate':
        return 0 if result['passed'] else 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
