import numpy as np
from PIL import Image
import pytest
import torch

from dual_payload.channel import TransmissionChannel


def test_identity_is_unclamped_and_differentiable():
    x = torch.tensor([[[[-0.1, 0.5, 1.1]]]], requires_grad=True)
    xq, result = TransmissionChannel()(x, {})
    assert xq is x and result.attacked_image is x
    result.attacked_image.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_ste_matches_real8_and_clamp_gradient():
    x = torch.tensor([[[[-0.1, 0.2345, 1.1]]]], requires_grad=True)
    ste = TransmissionChannel('ste8', True)
    output = ste.quantize(x)
    with torch.no_grad():
        real = TransmissionChannel('real8', True).quantize(x)
    torch.testing.assert_close(output, real, atol=1e-7, rtol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[[0., 1., 0.]]]]))


def test_real8_matches_png_uint8_roundtrip(tmp_path):
    x = torch.rand(1, 1, 16, 16)
    quantized = TransmissionChannel('real8', True).quantize(x)
    path = tmp_path / 'carrier.png'
    Image.fromarray((x[0, 0].numpy() * 255).round().astype(np.uint8)).save(path)
    with Image.open(path) as saved:
        loaded = torch.from_numpy(np.array(saved, dtype=np.float32) / 255)[None, None]
    assert torch.equal(quantized, loaded)


def test_invalid_channel_modes_fail_explicitly():
    with pytest.raises(ValueError):
        TransmissionChannel('ste8', False)
    with pytest.raises(NotImplementedError):
        TransmissionChannel(attack_mode='jpeg')
    with pytest.raises(RuntimeError):
        TransmissionChannel('real8', True).quantize(torch.ones(1, 1, 8, 8, requires_grad=True))
