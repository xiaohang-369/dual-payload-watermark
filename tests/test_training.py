from copy import deepcopy
import json

from PIL import Image
import pytest
import torch

from dual_payload.config import DEFAULT_CONFIG
from dual_payload.system import DualPayloadSystem
from dual_payload.training import (evaluate_full_system_bank, evaluate_main, load_checkpoint,
                                   train_main)


def test_resume_matches_uninterrupted_cpu_training(tmp_path):
    config = tmp_path / 'tiny.json'
    config.write_text(json.dumps({'model': {'channels': 4, 'blocks': 1}}), encoding='utf-8')
    interrupted, uninterrupted = tmp_path / 'interrupted', tmp_path / 'uninterrupted'
    base = ['--config', str(config), '--smoke', '--device', 'cpu']
    train_main(base + ['--output-dir', str(interrupted), '--max-steps', '2'])
    train_main(['--resume', str(interrupted / 'last.pt'), '--max-steps', '3', '--device', 'cpu'])
    train_main(base + ['--output-dir', str(uninterrupted), '--max-steps', '3'])
    resumed = load_checkpoint(interrupted / 'last.pt')
    straight = load_checkpoint(uninterrupted / 'last.pt')
    assert resumed['global_step'] == straight['global_step'] == 3
    for key, value in resumed['model'].items():
        torch.testing.assert_close(value, straight['model'][key], atol=0, rtol=0)
    assert resumed['validation'] == straight['validation']
    assert (interrupted / 'preview.png').is_file()
    assert (interrupted / 'best.pt').is_file()
    with pytest.raises(ValueError, match='not empty'):
        train_main(base + ['--output-dir', str(interrupted)])
    with pytest.raises(ValueError, match='cannot change'):
        train_main(['--resume', str(interrupted / 'last.pt'), '--max-steps', '4', '--image-size', '16'])
    evaluate_main(['--checkpoint', str(interrupted / 'last.pt'), '--smoke', '--device', 'cpu',
                   '--quantization-mode', 'real8', '--output', str(tmp_path / 'evaluation.json')])
    report = json.loads((tmp_path / 'evaluation.json').read_text(encoding='utf-8'))
    assert report['channel']['quantization_mode'] == 'real8'
    assert report['synthetic'] is True


def test_no_silent_synthetic_training():
    with pytest.raises(SystemExit) as error:
        train_main([])
    assert error.value.code == 2


def test_full_system_bank_is_no_grad_and_preserves_every_parameter():
    config = deepcopy(DEFAULT_CONFIG)
    config['model'].update(channels=4, blocks=1)
    model = DualPayloadSystem(config['model'], config['channel'])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    messages = torch.randint(0, 2, (3, 64)).float()
    result = evaluate_full_system_bank(
        model, [torch.rand(3, 16, 16), torch.rand(3, 16, 16)], messages,
        torch.device('cpu'), batch_size=2)
    assert result['pairs'] == 6
    assert result['watermark']['bits'] == 6 * 64
    assert set(result['color_only']['psnr']) == {'mean', 'min', 'max'}
    assert set(result['full']['ssim']) == {'mean', 'min', 'max'}
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_system_manifest_cli_uses_saved_images_and_messages(tmp_path, capsys):
    config = deepcopy(DEFAULT_CONFIG)
    config['model'].update(channels=4, blocks=1)
    config['data'].update(image_size=16, batch_size=2, num_workers=0)
    model = DualPayloadSystem(config['model'], config['channel'])
    messages = torch.randint(0, 2, (2, 64)).float()
    image_paths = []
    for index, color in enumerate(((40, 80, 120), (120, 80, 40))):
        path = tmp_path / f'image_{index}.png'
        Image.new('RGB', (24, 16), color).save(path)
        image_paths.append(str(path.resolve()))
    checkpoint_path = tmp_path / 'diagnostic_weights.pt'
    torch.save({'diagnostic_format_version': 1, 'kind': 'watermark_only_bce_overfit',
                'model': model.state_dict(), 'source_config': config,
                'diagnostic_steps': 7, 'fit_messages': messages,
                'pair_exposure_counts': torch.ones(4, dtype=torch.long)}, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    manifest = {'mode': 'overfit', 'source_config': config, 'image_size': 16,
                'crop': 'fixed_center', 'images': image_paths,
                'fit_messages': messages.int().tolist(), 'fit_message_count': 2,
                'cli': {'steps': 7}, 'eval_batch_size': 2}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    output = tmp_path / 'full_system_report.json'

    evaluate_main(['--checkpoint', str(checkpoint_path), '--manifest', str(manifest_path),
                   '--device', 'cpu', '--output', str(output)])

    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['mode'] == 'full_system_baseline'
    assert report['image_count'] == 2
    assert report['message_count'] == 2
    assert report['pairs'] == 4
    assert report['watermark']['bits'] == 256
    assert report['messages'] == manifest['fit_messages']
    assert report['pair_order'] == 'image_major_cartesian'
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    terminal = capsys.readouterr().out
    for heading in ('Watermark:', 'Color-only:', 'Full:', 'Impact:', 'Carrier:'):
        assert heading in terminal
