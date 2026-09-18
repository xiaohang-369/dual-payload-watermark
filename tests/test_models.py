import pytest
import torch

from dual_payload.config import DEFAULT_CONFIG
from dual_payload.losses import CleanLoss
from dual_payload.models import ColorEncoder, WatermarkEncoder, WatermarkDecoder
from dual_payload.system import DualPayloadSystem
from dual_payload.transforms import BlockDCT, rgb_to_ycbcr


def tiny_system():
    return DualPayloadSystem({'channels': 8, 'blocks': 1})


def test_zero_heads_make_initial_joint_carrier_equal_luma():
    model = tiny_system()
    rgb, message = torch.rand(2, 3, 16, 24), torch.randint(0, 2, (2, 64)).float()
    output = model(rgb, message)
    assert output['logits'].shape == (2, 64)
    assert output['rgb_hat'].shape == rgb.shape
    assert torch.equal(output['x_float'], output['y'])
    assert output['attacked_image'] is output['x_float']
    assert output['target_rgb'] is rgb
    assert bool((output['valid_mask'] == 1).all())
    assert int(torch.count_nonzero(model.color_decoder.luma_head.weight)) > 0
    assert int(torch.count_nonzero(model.watermark_decoder.head.weight)) > 0


def test_both_decoders_receive_same_image_and_keep_p0_constraint():
    model = tiny_system()
    inputs = []
    handles = [decoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
               for decoder in (model.color_decoder, model.watermark_decoder)]
    output = model(torch.rand(1, 3, 16, 16), torch.zeros(1, 64))
    assert inputs[0] is output['attacked_image'] and inputs[1] is inputs[0]
    dct = BlockDCT()
    torch.testing.assert_close(dct.project(output['y_hat'], '0'),
                               dct.project(output['attacked_image'], '0'), atol=3e-6, rtol=2e-6)
    for handle in handles:
        handle.remove()


def test_first_step_encoder_trunk_is_zero_gradient_then_receives_gradient():
    torch.manual_seed(9)
    encoder = ColorEncoder(channels=8, blocks=1)
    rgb = torch.rand(2, 3, 16, 16)
    y, cb, cr = rgb_to_ycbcr(rgb)
    target = torch.rand_like(y)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    output = encoder(y, cb, cr)
    (output['carrier'] - target).square().mean().backward()
    assert encoder.head.weight.grad.abs().sum() > 0
    assert encoder.stem.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = encoder(y, cb, cr)
    (output['carrier'] - target).square().mean().backward()
    assert encoder.stem.weight.grad.abs().sum() > 0


def test_full_loss_has_finite_gradients_for_all_four_heads():
    torch.manual_seed(31)
    model = tiny_system()
    message = torch.randint(0, 2, (2, 64)).float()
    losses = CleanLoss(DEFAULT_CONFIG['loss'])(model(torch.rand(2, 3, 16, 16), message), message)
    losses['total'].backward()
    for layer in (model.color_encoder.head, model.watermark_encoder.head,
                  model.color_decoder.chroma_head, model.watermark_decoder.head):
        assert layer.weight.grad is not None
        assert bool(torch.isfinite(layer.weight.grad).all())
        assert layer.weight.grad.abs().sum() > 0


def test_decoder_has_raw_logits_not_sigmoid():
    decoder = WatermarkDecoder(channels=8, blocks=1)
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.fill_(-3)
    assert torch.equal(decoder(torch.rand(1, 1, 16, 16)), torch.full((1, 64), -3.0))


@pytest.mark.parametrize('message', [torch.zeros(1, 63), torch.full((1, 64), 0.5)])
def test_bad_message_rejected(message):
    with pytest.raises(ValueError):
        WatermarkEncoder(channels=8, blocks=1)(torch.rand(1, 1, 16, 16), message)


def test_clean_algebra_separates_nonzero_payloads():
    dct = BlockDCT()
    y = torch.rand(2, 1, 16, 16)
    dc, dw = dct.project(torch.randn_like(y), 'c'), dct.project(torch.randn_like(y), 'w')
    x, s = y + dc + dw, y + dc
    torch.testing.assert_close(x - dct.project(x, 'w'), s - dct.project(s, 'w'), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(dct.watermark(x), dct.watermark(y + dw), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(s - dct.project(s, 'c'), y - dct.project(y, 'c'), atol=2e-6, rtol=2e-6)
