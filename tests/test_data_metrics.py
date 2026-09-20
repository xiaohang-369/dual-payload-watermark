import json

import numpy as np
from PIL import Image
import pytest
import torch

from dual_payload.config import architecture_version, load_config
from dual_payload.data import ImageFolderDataset, ensure_disjoint, fixed_message, load_rgb_image
from dual_payload.metrics import (compute_metrics, psnr, psnr_per_sample, ssim,
                                  ssim_per_sample)


def test_folder_rgb_range_and_deterministic_validation(tmp_path):
    image = np.arange(24 * 40 * 3, dtype=np.uint8).reshape(24, 40, 3)
    Image.fromarray(image).save(tmp_path / 'image.png')
    dataset = ImageFolderDataset(tmp_path, image_size=16)
    sample = dataset[0]
    assert sample['rgb'].shape == (3, 16, 16)
    assert sample['rgb'].dtype == torch.float32
    assert 0 <= float(sample['rgb'].min()) <= float(sample['rgb'].max()) <= 1
    assert torch.equal(sample['rgb'], dataset[0]['rgb'])
    assert torch.equal(sample['rgb'], load_rgb_image(tmp_path / 'image.png', 16))
    assert torch.equal(sample['message'], dataset[0]['message'])
    with pytest.raises(ValueError, match='overlap'):
        ensure_disjoint(dataset.paths, dataset.paths)
    assert not torch.equal(fixed_message(0, 2026), fixed_message(1, 2026))


def test_training_crops_reproducible_for_resume(tmp_path):
    Image.fromarray(np.arange(16 * 64 * 3, dtype=np.uint8).reshape(16, 64, 3)).save(tmp_path / 'image.png')
    dataset = ImageFolderDataset(tmp_path, image_size=16, training=True)
    dataset.set_epoch(4)
    a = dataset[0]['rgb']
    torch.rand(100)  # Unrelated RNG use must not alter the crop.
    assert torch.equal(a, dataset[0]['rgb'])


def test_metrics_identity_and_bit_accounting():
    image = torch.rand(2, 3, 16, 16)
    assert float(psnr(image, image)) == 120
    assert psnr_per_sample(image, image).tolist() == [120, 120]
    torch.testing.assert_close(ssim(image, image), torch.tensor(1.))
    torch.testing.assert_close(ssim_per_sample(image, image), torch.ones(2))
    message = torch.zeros(2, 64)
    y = image.mean(dim=1, keepdim=True)
    output = {
        'attack_info': {'type': 'identity'}, 'valid_mask': torch.ones_like(y),
        'logits': torch.full((2, 64), -1.), 'rgb_hat': image,
        'target_rgb': image, 'x_float': y, 'x_quantized': y, 'y': y,
        'delta_c': torch.zeros_like(y), 'delta_w': torch.zeros_like(y),
    }
    output['logits'][0, 0] = 0  # zero must decode to 1
    metrics = compute_metrics(output, message)
    assert float(metrics['ber']) == 1 / 128
    assert float(metrics['bit_accuracy']) == 1 - 1 / 128
    assert float(metrics['message_accuracy']) == 0.5


@pytest.mark.parametrize('override', [
    {'model': {'mystery': 1}}, {'data': {'image_size': 15}},
    {'channel': {'clamp_enabled': 'false'}}, {'loss': {'rgb': -1}},
    {'model': {'architecture_version': 'v2', 'channels': 64}},
    {'model': {'architecture_version': 'v2'}, 'data': {'image_size': 32}},
])
def test_invalid_config_fails(tmp_path, override):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(override), encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_architecture_version_is_classified_as_v1(tmp_path):
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps({'model': {'channels': 64, 'blocks': 8}}), encoding='utf-8')
    assert architecture_version(load_config(path)) == 'v1'
    assert architecture_version(load_config()) == 'v2'
