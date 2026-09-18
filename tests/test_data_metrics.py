import json

import numpy as np
from PIL import Image
import pytest
import torch

from dual_payload.config import load_config
from dual_payload.data import ImageFolderDataset, ensure_disjoint, fixed_message, load_rgb_image
from dual_payload.metrics import (compute_metrics, psnr, psnr_per_sample, ssim,
                                  ssim_per_sample)
from dual_payload.system import DualPayloadSystem


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
    model = DualPayloadSystem({'channels': 8, 'blocks': 1})
    message = torch.zeros(2, 64)
    output = model(image, message)
    output['logits'] = torch.full((2, 64), -1.)
    output['logits'][0, 0] = 0  # zero must decode to 1
    metrics = compute_metrics(output, message)
    assert float(metrics['ber']) == 1 / 128
    assert float(metrics['bit_accuracy']) == 1 - 1 / 128
    assert float(metrics['message_accuracy']) == 0.5


@pytest.mark.parametrize('override', [
    {'model': {'mystery': 1}}, {'data': {'image_size': 15}},
    {'channel': {'clamp_enabled': 'false'}}, {'loss': {'rgb': -1}},
])
def test_invalid_config_fails(tmp_path, override):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(override), encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)
