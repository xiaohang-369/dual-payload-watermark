"""Synthetic program checks only; no claim of trained medical performance."""
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dual_payload.medical.artifacts import load_record, sha256
from dual_payload.medical.calibration import calibrate
from dual_payload.medical.data import create_manifest, MedicalDataset, assign_pad_splits, audit_records
from dual_payload.medical.evaluation import (export_experiment, evaluation_policy_template,
                                            evaluate_export, summarize, image_metrics)
from dual_payload.medical.learning import (ResidualQuantizer, PayloadFactory, initialize_models,
                                          make_models, state_digest, experiment_forward)
from dual_payload.medical.profile import read_json
from dual_payload.medical.protocol import dequantize_residual
from dual_payload.medical.training import training_template, train, validate_training, STAGES


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def picture(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (24, 40, 3), dtype=np.uint8)).save(path)


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp('experiment')
    pad, derm = root / 'pad', root / 'derm'
    pad.mkdir(); (derm / 'meta').mkdir(parents=True)
    rows = []
    for i in range(9):
        picture(pad / f'p{i}.png', i)
        rows.append(dict(patient_id=f'p{i}', lesion_id=f'l{i}', img_id=f'p{i}.png', diagnostic='synthetic'))
    write_csv(pad / 'metadata.csv', rows[0].keys(), rows)
    drows = []
    for i in range(2):
        for modality in ('clinic', 'derm'):
            picture(derm / 'images' / f'{i}-{modality}.png', 20 + 2*i + (modality == 'derm'))
        drows.append(dict(case_num=str(i), clinic=f'{i}-clinic.png', derm=f'{i}-derm.png', diagnosis='synthetic'))
    write_csv(derm / 'meta' / 'meta.csv', drows[0].keys(), drows)
    manifest = root / 'manifest.json'
    create_manifest(pad, pad / 'metadata.csv', manifest, (0.6, 0.2, 0.2), 2026,
                    derm, derm / 'meta' / 'meta.csv')
    limits = {'color_residual': 2/255, 'color_ciphertext': 2/255, 'patient_ciphertext': 2/255}
    initialization = {'kind': 'scratch', 'path': None, 'sha256': None}
    roots = {'pad': str(pad), 'derm7pt': str(derm)}
    training = MedicalDataset(manifest, roots, 'train')
    calibrated = root / 'calibration.json'
    calibrate(training, initialization, limits, calibrated, quantile=0.999, minimum_rms=1e-8)
    return dict(root=root, pad=pad, derm=derm, manifest=manifest, roots=roots,
                limits=limits, initialization=initialization, calibration=calibrated)


def config_for(assets, output):
    config = training_template()
    config.update(purpose='test', device='cpu', stage='transport', manifest=str(assets['manifest']),
                  roots=assets['roots'], calibration=str(assets['calibration']),
                  initialization=assets['initialization'], rms_limits=assets['limits'],
                  contract={'profile_id': 65001, 'quantization_id': 65001,
                            'interleaver_seeds': {'color': 100, 'patient': 101}},
                  epochs=2, max_steps=2, batch_size=2, micro_batch_size=1, lr=1e-4,
                  save_every=1, loss_weights={'color_bits': 1., 'patient_bits': 1., 'gray': 1., 'range': 0.1},
                  output=str(output))
    return config


def test_manifest_grouping_external_cohorts_and_shared_preprocessing(assets):
    document = load_record(assets['manifest'])
    assert len(document['records']) == 13
    external = MedicalDataset(assets['manifest'], assets['roots'], 'external')
    assert len(external) == 4
    assert {r['modality'] for r in external.records} == {'clinical', 'dermoscopic'}
    assert external[0]['rgb'].shape == (3, 256, 256)
    assert external[0]['content_rect'].tolist() == [0, 51, 256, 154]
    all_ids = []
    for split in ('train', 'valid', 'test'):
        ds = MedicalDataset(assets['manifest'], assets['roots'], split)
        all_ids.extend(r['patient_id'] for r in ds.records)
    assert len(set(all_ids)) == 9
    broken = deepcopy(document['records'])
    next(r for r in broken if r['dataset'] == 'derm7pt')['split'] = 'train'
    with pytest.raises(ValueError):
        audit_records(broken)


def test_duplicates_merge_groups_before_splitting_and_cross_split_is_rejected():
    records = [dict(dataset='pad', image_id=str(i), patient_id=str(i), case_id=str(i),
                    file_sha256=str(i), work_sha256='duplicate' if i < 2 else str(i)) for i in range(5)]
    assign_pad_splits(records, (0.6, 0.2, 0.2), 7)
    assert records[0]['split'] == records[1]['split']
    audit_records(records)
    records[1]['split'] = 'test' if records[0]['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='leakage'):
        audit_records(records)


def test_manifest_detects_file_mutation(assets, tmp_path):
    import shutil
    pad = tmp_path / 'pad'; shutil.copytree(assets['pad'], pad)
    ds = MedicalDataset(assets['manifest'], dict(assets['roots'], pad=str(pad)), 'train')
    file = pad / ds.records[0]['path']
    file.write_bytes(file.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        ds[0]


def test_calibration_is_train_only_and_rejects_zero_initialization(assets, tmp_path, monkeypatch):
    validation = MedicalDataset(assets['manifest'], assets['roots'], 'valid')
    with pytest.raises(ValueError, match='training split'):
        calibrate(validation, assets['initialization'], assets['limits'], tmp_path/'bad.json',
                  quantile=0.999, minimum_rms=1e-8)
    training = MedicalDataset(assets['manifest'], assets['roots'], 'train')
    import dual_payload.medical.calibration as module
    zero = make_models(assets['limits'], seed=2026)
    with torch.no_grad():
        zero['ec'].head.weight.zero_()
        zero['ec'].head.bias.zero_()
    monkeypatch.setattr(module, 'make_models', lambda *args, **kwargs: zero)
    with pytest.raises(ValueError, match='degenerate'):
        calibrate(training, {'kind': 'scratch', 'path': None, 'sha256': None}, assets['limits'],
                  tmp_path/'zero.json', quantile=0.999, minimum_rms=1e-8)
    result = load_record(assets['calibration'])
    assert len(result['steps']) == 39 and result['residual_rms'] > 0
    assert result['manifest_sha256'] == training.manifest['sha256']


def test_ste_matches_wire_reconstruction_and_keeps_gradients(assets):
    steps = load_record(assets['calibration'])['steps']
    quantizer = ResidualQuantizer(steps)
    residual = (torch.randn(2, 1, 256, 256) * 0.01).requires_grad_()
    out = quantizer(residual)
    for i in range(2):
        q = out['integers'][i].permute(1, 2, 0).numpy().astype(np.int8)
        expected = dequantize_residual(q, steps)
        torch.testing.assert_close(out['residual'][i:i+1], expected, atol=5e-8, rtol=1e-5)
    out['residual'].square().mean().backward()
    assert residual.grad.abs().sum() > 0 and torch.isfinite(residual.grad).all()


def test_training_config_is_separate_from_final_profile(assets, tmp_path):
    config = config_for(assets, tmp_path/'run')
    assert 'weights' not in config
    validate_training(config, allow_test=True)
    with pytest.raises(ValueError, match='allow_test'):
        validate_training(config)
    with pytest.raises(ValueError):
        validate_training(training_template())
    wrong = deepcopy(config); wrong['loss_weights']['patient_bits'] = 0
    with pytest.raises(ValueError, match='Both payload'):
        validate_training(wrong, allow_test=True)
    with pytest.raises(ValueError, match='RMS limits'):
        make_models(training_template()['rms_limits'])
    with pytest.raises(ValueError, match='must not specify checkpoint'):
        initialize_models(make_models(assets['limits']),
                          {'kind': 'scratch', 'path': 'old.pt', 'sha256': '0'*64})
    with pytest.raises(ValueError, match='scratch or a matching'):
        initialize_models(make_models(assets['limits']),
                          {'kind': 'v2', 'path': 'old.pt', 'sha256': '0'*64})


@pytest.fixture(scope='module')
def trained(assets):
    output = assets['root'] / 'training-run'
    config = config_for(assets, output)
    first = train(config, allow_test=True, stop_after=1)
    assert first['step'] == 1
    last = train(config, allow_test=True, resume=first['checkpoint'], resume_sha256=first['sha256'])
    assert last['step'] == 2
    return config, last


def test_training_real_frames_validation_resume_and_frozen_ec(assets, trained):
    config, last = trained
    state = torch.load(last['checkpoint'], map_location='cpu', weights_only=True)
    models = make_models(assets['limits']); models.load_state_dict(state['models'])
    assert state_digest(models['ec']) == load_record(assets['calibration'])['ec_state_sha256']
    assert state['optimizer']['state'] and state['next_batch'] == 2
    assert (Path(config['output']) / 'best.pt').is_file()
    assert list((Path(config['output']) / 'validation_frames').glob('*.json'))
    assert 0 <= last['validation']['color_bits_ber'] <= 1
    assert 0 <= last['validation']['patient_bits_ber'] <= 1
    with pytest.raises(ValueError, match='SHA-256'):
        train(config, allow_test=True, resume=last['checkpoint'], resume_sha256='0'*64)


@pytest.mark.parametrize('stage', ['color', 'decoder', 'joint'])
def test_other_stage_backward_reaches_only_selected_networks(assets, tmp_path, stage):
    config = config_for(assets, tmp_path/'run')
    config['stage'] = stage
    config['loss_weights'] = {'rgb': 1.}
    if stage == 'joint': config['loss_weights'].update(color_bits=1., patient_bits=1., gray=1., range=0.1)
    models = make_models(assets['limits'], seed=config['seed']); initialize_models(models, assets['initialization'])
    for name, module in models.items(): module.requires_grad_(name in STAGES[stage])
    quantizer = ResidualQuantizer(load_record(assets['calibration'])['steps'])
    payload = PayloadFactory(config['contract'], tmp_path/'nonce.sqlite', 'cpu')
    sample = MedicalDataset(assets['manifest'], assets['roots'], 'train')[0]
    batch = {'rgb': sample['rgb'][None], 'content_rect': sample['content_rect'][None], 'image_id': [sample['image_id']]}
    output = experiment_forward(models, quantizer, payload, batch, config)
    output['total'].backward()
    for name, module in models.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if name in STAGES[stage]:
            assert gradients and any(g.abs().sum() > 0 for g in gradients)
        else:
            assert not gradients


def test_fixed_validation_bank_is_reused_without_new_encryption(assets, tmp_path):
    config = config_for(assets, tmp_path/'run')
    q = torch.zeros(1, 39, 32, 32)
    factory = PayloadFactory(config['contract'], tmp_path/'nonce.sqlite', 'cpu')
    c1, m1 = factory.make(q, ['sample'], 1, validation=True)
    factory2 = PayloadFactory(config['contract'], tmp_path/'nonce.sqlite', 'cpu')
    factory2.nonces.reserve = lambda key: (_ for _ in ()).throw(AssertionError('bank must reuse frames'))
    c2, m2 = factory2.make(q, ['sample'], 1, validation=True)
    assert torch.equal(c1,c2) and torch.equal(m1,m2)


def test_freeze_and_real_png_batch_evaluation_in_separate_process(assets, trained, tmp_path):
    config, last = trained
    policy = evaluation_policy_template()
    policy['thresholds'] = {name: 0. for name in policy['thresholds']}  # Test fixture only.
    export = tmp_path/'export'
    metadata = export_experiment(last['checkpoint'], last['sha256'], export, policy, allow_test=True,
                                 profile_id=65002)
    assert metadata['weights'] and metadata['profile_sha256']
    assert metadata['profile_path'] == 'profiles/65002'
    summary = evaluate_export(export, assets['manifest'], assets['roots'], 'test', tmp_path/'evaluation', allow_test=True)
    assert summary['overall']['samples'] == 2
    assert summary['overall']['authentication_success'] == 1
    assert summary['overall']['color_packet_success'] == 0  # Two steps are not reliable transmission.
    assert summary['overall']['rgb_integer'] is None
    assert summary['overall']['gray_samples'] == 2
    assert summary['overall']['color_ber_samples'] == 2
    assert summary['overall']['color_decoded_samples'] == 2
    assert len(summary['overall']['failures']) == 2
    assert not summary['overall']['passed']
    for root in (tmp_path/'evaluation').glob('*-received'):
        assert (root/'diagnostics.npz').is_file()


def test_failure_denominators_and_metric_regions():
    rows = [dict(image_id=str(i), authentication='AUTH_FAILED', color_ok=False, patient_ok=False) for i in range(3)]
    policy = evaluation_policy_template(); limits = {name: 0. for name in policy['thresholds']}
    report = summarize(rows, limits)
    assert report['samples'] == 3 and report['joint_success'] == 0
    assert report['color_ber'] is None and report['rgb_float_samples'] == 0
    image = torch.zeros(1,3,32,32); target = image.clone(); image[:,:,0] = 1
    metrics = image_metrics(image, target, (0,1,32,31))
    assert metrics['content']['mse'] == 0 and metrics['whole']['mse'] > 0
    assert metrics['content']['delta_e76'] == 0


def test_metadata_missing_views_unsafe_paths_and_official_indexes(tmp_path):
    from dual_payload.medical.data import derm_records
    root = tmp_path/'derm'; (root/'meta').mkdir(parents=True)
    picture(root/'images'/'a.png', 10)
    picture(root/'images'/'b.png', 11)
    rows = [dict(case_num='0', clinic='', derm='a.png', diagnosis='synthetic'),
            dict(case_num='1', clinic='b.png', derm='', diagnosis='synthetic')]
    path = root/'meta'/'meta.csv'; write_csv(path, rows[0].keys(), rows)
    records, missing = derm_records(root, path)
    assert len(records) == 2 and len(missing) == 2
    rows[0]['derm'] = '../../outside.png'; write_csv(path, rows[0].keys(), rows)
    with pytest.raises(ValueError, match='relative'):
        derm_records(root, path)
    rows[0]['derm'] = 'a.png'; write_csv(path, rows[0].keys(), rows)
    write_csv(root/'meta'/'train_indexes.csv', ['indexes'], [{'indexes': 0}])
    with pytest.raises(ValueError, match='Incomplete'):
        derm_records(root, path)


def test_validation_config_and_calibration_artifacts_cannot_silently_change(assets, tmp_path):
    config = config_for(assets, tmp_path/'run')
    wrong = deepcopy(config); wrong['rms_limits']['color_residual'] *= 2
    with pytest.raises(ValueError, match='RMS limit differs'):
        train(wrong, allow_test=True, stop_after=1)
    calibrated = read_json(assets['calibration']); calibrated['steps'][0] *= 2
    import json
    path = tmp_path/'changed.json'; path.write_text(json.dumps(calibrated))
    with pytest.raises(ValueError, match='checksum'):
        load_record(path)


def test_frozen_evaluation_thresholds_are_required(assets, trained, tmp_path):
    _, last = trained
    with pytest.raises(ValueError, match='threshold'):
        export_experiment(last['checkpoint'], last['sha256'], tmp_path/'export',
                          evaluation_policy_template(), allow_test=True)
    assert not (tmp_path/'export').exists()


def test_h100_optimizer_groups_and_resumable_schedule(assets, tmp_path):
    from dual_payload.medical.main_experiment import h100_config
    from dual_payload.medical.optimization import make_optimizer, schedule_step
    config = h100_config()
    config.update(manifest=str(assets['manifest']), roots=assets['roots'],
                  calibration=str(assets['calibration']), initialization=assets['initialization'],
                  output=str(tmp_path/'run'))
    config['contract'].update(profile_id=1, quantization_id=1)
    validate_training(config)
    models = make_models(assets['limits'])
    initialize_models(models, assets['initialization'])
    for name, model in models.items(): model.requires_grad_(name in ('ec', 'dc'))
    optimizer = make_optimizer(models, config)
    assert isinstance(optimizer, torch.optim.AdamW)
    groups = {g['group_name']: g for g in optimizer.param_groups}
    assert set(groups) == {'all'} and groups['all']['lr'] == 1e-4
    trained_ids = {id(p) for p in groups['all']['params']}
    assert id(models['dc'].gray_stem.weight) in trained_ids
    assert id(models['dc'].chroma_head.weight) in trained_ids
    schedule_step(optimizer, config, 199, 1000)
    assert groups['all']['lr'] == pytest.approx(1e-4)
    schedule_step(optimizer, config, 500, 1000)
    saved = deepcopy(optimizer.state_dict())
    resumed = make_optimizer(models, config); resumed.load_state_dict(saved)
    schedule_step(optimizer, config, 501, 1000)
    schedule_step(resumed, config, 501, 1000)
    assert [g['lr'] for g in resumed.param_groups] == [g['lr'] for g in optimizer.param_groups]
    schedule_step(optimizer, config, 999, 1000)
    assert all(g['lr'] == pytest.approx(1e-6) for g in optimizer.param_groups)


def test_geometry_preserves_pixels_and_content_region():
    from dual_payload.medical.optimization import transform_geometry, augment_batch
    image = torch.zeros(3, 256, 256)
    image[:, 17:58, 25:64] = torch.arange(39*41).reshape(1, 41, 39) + 1
    rect = (25, 17, 39, 41)
    for code in range(16):
        transformed, (x, y, w, h) = transform_geometry(image, rect, code)
        assert int((transformed[0] > 0).sum()) == w*h == 39*41
        assert (transformed[:, y:y+h, x:x+w] > 0).all()
        assert torch.equal(transformed.flatten().sort().values, image.flatten().sort().values)
    batch = {'rgb': image[None], 'content_rect': torch.tensor([rect]), 'image_id': ['pad:0']}
    first = augment_batch(batch, 2026, 3)
    repeated = augment_batch(batch, 2026, 3)
    assert torch.equal(first['rgb'], repeated['rgb']) and torch.equal(batch['rgb'], image[None])


def test_adamw_augmented_resume_matches_uninterrupted_color_training(assets, tmp_path):
    from dual_payload.medical.main_experiment import h100_config, run_training
    config = config_for(assets, tmp_path/'split')
    preset = h100_config()
    config.update(stage='color', lr=1e-5, loss_weights={'rgb': 1.0},
                  optimization=preset['optimization'], augmentation='dihedral', weight_decay=1e-4)
    config['optimization']['warmup_steps'] = 1
    first = train(config, allow_test=True, stop_after=1)
    train(config, allow_test=True, resume=first['checkpoint'], resume_sha256=first['sha256'])
    uninterrupted = deepcopy(config); uninterrupted['output'] = str(tmp_path/'whole')
    train(uninterrupted, allow_test=True)
    left = torch.load(tmp_path/'split'/'last.pt', weights_only=True)
    right = torch.load(tmp_path/'whole'/'last.pt', weights_only=True)
    for key in left['models']:
        torch.testing.assert_close(left['models'][key], right['models'][key], rtol=0, atol=0)
    # Stage receipts keep checkpoint digests separate from record checksums.
    result = run_training(config)
    assert result['checkpoint_sha256'] == sha256(result['checkpoint'])
    assert run_training(config) == result


def test_main_joint_run_uses_one_calibration_and_updates_all_networks(assets, tmp_path, monkeypatch):
    import dual_payload.medical.main_experiment as main
    preset = main.h100_config
    def small():
        config = preset()
        config.update(device='cpu', epochs=1, max_steps=1, batch_size=2)
        config['optimization']['warmup_steps'] = 0
        return config
    monkeypatch.setattr(main, 'h100_config', small)
    result = main._run_locked(assets['pad'], assets['pad']/'metadata.csv', tmp_path, 1, 1, 1)
    assert result['status'] == 'training_complete'
    config = read_json(tmp_path/'train-joint.json')
    assert config['stage'] == 'joint' and config['augmentation'] == 'none'
    assert config['initialization'] == {'kind': 'scratch', 'path': None, 'sha256': None}
    assert config['lr'] == 1e-4 and 'new_layer_lr' not in config['optimization']
    assert len(list(tmp_path.glob('calibration*.json'))) == 1
    assert not list(tmp_path.glob('stage-*'))
    torch.manual_seed(config['seed'])
    initial = make_models(config['rms_limits'], seed=config['seed']); initialize_models(initial, config['initialization'])
    trained = torch.load(tmp_path/'joint'/'last.pt', weights_only=True)['models']
    for name in ('ec', 'ew', 'dc', 'dw'):
        assert any(not torch.equal(value, trained[name+'.'+key]) for key, value in initial[name].state_dict().items())
    assert main._run_locked(assets['pad'], assets['pad']/'metadata.csv', tmp_path, 1, 1, 1) == result


def test_scratch_seed_is_independent_of_ambient_rng_and_matches_calibration(assets, tmp_path):
    torch.manual_seed(8)
    before = torch.get_rng_state().clone()
    first = make_models(assets['limits'], seed=2026)
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(100)
    second = make_models(assets['limits'], seed=2026)
    for name in first:
        assert state_digest(first[name]) == state_digest(second[name])
    assert state_digest(first['ec']) == load_record(assets['calibration'])['ec_state_sha256']
    assert state_digest(make_models(assets['limits'], seed=2027)['ec']) != state_digest(first['ec'])
    wrong = config_for(assets, tmp_path/'wrong-seed'); wrong['seed'] += 1
    with pytest.raises(ValueError, match='seed differs'):
        train(wrong, allow_test=True)
    assert not Path(wrong['output']).exists()


def test_initial_checkpoint_recovers_interruption_before_first_update(assets, tmp_path, monkeypatch):
    import dual_payload.medical.training as module
    config = config_for(assets, tmp_path/'interrupted')
    config.update(stage='joint', max_steps=1,
                  loss_weights={'rgb': 1., 'color_bits': 1., 'patient_bits': 1., 'gray': 1., 'range': .1})
    original_forward = module.experiment_forward
    def interrupted(*args, **kwargs):
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(module, 'experiment_forward', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(config, allow_test=True)
    checkpoint = Path(config['output'])/'last.pt'
    state = torch.load(checkpoint, weights_only=True)
    assert state['step'] == 0 and state['best'] is None and not state['optimizer']['state']
    monkeypatch.setattr(module, 'experiment_forward', original_forward)
    result = train(config, allow_test=True, resume=str(checkpoint), resume_sha256=sha256(checkpoint))
    assert result['step'] == 1
    assert (Path(config['output'])/'best.pt').is_file()


def test_main_cli_requires_only_data_and_output(monkeypatch, tmp_path):
    import dual_payload.medical.main_experiment as main
    observed = []
    monkeypatch.setattr(main, 'run_main', lambda *args, **kwargs: observed.append((args, kwargs)) or {})
    main.main(['--pad-root', '/data/pad', '--pad-metadata', '/data/pad/metadata.csv',
               '--output', str(tmp_path)])
    assert observed[0][0] == ('/data/pad', '/data/pad/metadata.csv', str(tmp_path))


def test_checked_in_presets_match_entrypoints():
    from dual_payload.medical.main_experiment import h100_config
    configs = Path(__file__).resolve().parents[2]/'configs'
    assert read_json(configs/'medical_h100_80gb_joint.json') == h100_config()
    assert read_json(configs/'medical_train.template.json') == training_template()
