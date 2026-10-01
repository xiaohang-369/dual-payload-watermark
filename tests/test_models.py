"""Shared neural components retained by the medical training and receiver."""
import pytest
import torch

from dual_payload.models import (BiasFreeLayerNorm, ColorResidualEncoder, LayerNorm2d,
                                 RestormerUNet, WithBiasLayerNorm)
from dual_payload.medical.learning import make_models
from dual_payload.transforms import BlockDCT, rgb_to_ycbcr


def test_medical_state_dict_roundtrip_and_independent_networks():
    limits = {'color_residual': 2/255, 'color_ciphertext': 2/255, 'patient_ciphertext': 1/255}
    source = make_models(limits, seed=2026)
    restored = make_models(limits, seed=1)
    restored.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert all(layer.eps == 1e-5 for layer in source.modules()
               if isinstance(layer, (BiasFreeLayerNorm, WithBiasLayerNorm)))
    parameter_sets = [{p.data_ptr() for p in model.parameters()} for model in source.values()]
    for i, parameters in enumerate(parameter_sets):
        for other in parameter_sets[i+1:]:
            assert not parameters.intersection(other)


@pytest.mark.parametrize('bias', [False, True])
def test_channel_normalization_does_not_mix_spatial_positions(bias):
    layer = LayerNorm2d(8, bias)
    image = torch.randn(2, 8, 4, 6)
    changed = image.clone(); changed[:, :, 0, 0] += 5
    before, after = layer(image), layer(changed)
    torch.testing.assert_close(before[:, :, 1:, :], after[:, :, 1:, :], rtol=0, atol=0)
    assert torch.isfinite(after).all()


def test_shared_restormer_preserves_shape_and_input_gradients():
    model = RestormerUNet(layer_norm_bias=True)
    image = torch.randn(1, 24, 16, 24, requires_grad=True)
    output = model(image)
    assert output.shape == image.shape
    output.square().mean().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_shared_color_encoder_keeps_band_limit_and_immediate_trunk_gradient():
    torch.manual_seed(9)
    encoder = ColorResidualEncoder(residual_rms=2/255)
    result = encoder(*rgb_to_ycbcr(torch.rand(2, 3, 16, 24)))
    residual = result['residual']
    assert set(result) == {'candidate', 'residual'}
    assert residual.shape == (2, 1, 16, 24) and residual.abs().sum() > 0
    assert (residual.square().flatten(1).mean(1).sqrt() <= 2/255).all()
    torch.testing.assert_close(BlockDCT().project(residual, 'c'), residual, atol=1e-8, rtol=1e-4)
    residual.square().mean().backward()
    assert encoder.head.weight.grad.abs().sum() > 0
    assert encoder.stem.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('shape', [(1, 1, 15, 16), (1, 3, 16, 16)])
def test_shared_color_encoder_rejects_invalid_input(shape):
    encoder = ColorResidualEncoder(residual_rms=2/255)
    with pytest.raises(ValueError):
        encoder(*(torch.rand(shape) for _ in range(3)))
