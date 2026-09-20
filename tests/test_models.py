from copy import deepcopy

import pytest
import torch

from dual_payload.config import DEFAULT_CONFIG
from dual_payload.losses import CleanLoss
from dual_payload.models import (BiasFreeLayerNorm, ColorDecoder, ColorEncoder,
                                 WatermarkDecoder, WatermarkEncoder,
                                 WithBiasLayerNorm)
from dual_payload.system import DualPayloadSystem
from dual_payload.transforms import BlockDCT, rgb_to_ycbcr


def v2_system():
    return DualPayloadSystem(deepcopy(DEFAULT_CONFIG["model"]))


def test_missing_system_architecture_version_is_not_treated_as_v2():
    with pytest.raises(ValueError, match="architecture_version=v2"):
        DualPayloadSystem({})


def test_v2_construction_and_frozen_initialization():
    model = v2_system()
    assert model.architecture_version == "v2"
    assert not bool(torch.count_nonzero(model.color_encoder.head.weight))
    assert not bool(torch.count_nonzero(model.color_encoder.head.bias))
    assert 0 < float(model.watermark_encoder.head.weight.std()) < 5e-4
    assert not bool(torch.count_nonzero(model.watermark_encoder.head.bias))
    assert all(layer.eps == 1e-5 for layer in model.modules()
               if isinstance(layer, (BiasFreeLayerNorm, WithBiasLayerNorm)))
    ew_parameters = {parameter.data_ptr() for parameter in model.watermark_encoder.parameters()}
    dw_parameters = {parameter.data_ptr() for parameter in model.watermark_decoder.parameters()}
    assert not ew_parameters.intersection(dw_parameters)


def test_v2_state_dict_strict_roundtrip():
    source, restored = v2_system(), v2_system()
    restored.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_complete_system_forward_uses_fixed_v2_interface():
    model = v2_system()
    rgb = torch.rand(1, 3, 256, 256)
    message = torch.randint(0, 2, (1, 64)).float()
    with torch.no_grad():
        output = model(rgb, message)
    assert output["logits"].shape == (1, 64)
    assert output["rgb_hat"].shape == rgb.shape
    assert output["s"].shape == (1, 1, 256, 256)
    assert torch.equal(output["s"], output["y"])
    assert output["attacked_image"] is output["x_float"]
    assert output["target_rgb"] is rgb
    assert bool((output["valid_mask"] == 1).all())


def test_both_decoders_receive_same_256_image_and_keep_p0_constraint():
    model = v2_system()
    inputs = []
    handles = [decoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
               for decoder in (model.color_decoder, model.watermark_decoder)]
    with torch.no_grad():
        output = model(torch.rand(1, 3, 256, 256), torch.zeros(1, 64))
    assert inputs[0] is output["attacked_image"] and inputs[1] is inputs[0]
    dct = BlockDCT()
    torch.testing.assert_close(dct.project(output["y_hat"], "0"),
                               dct.project(output["attacked_image"], "0"),
                               atol=3e-6, rtol=2e-6)
    for handle in handles:
        handle.remove()


def test_ec_zero_head_delays_trunk_gradient_by_one_step():
    torch.manual_seed(9)
    encoder = ColorEncoder()
    rgb = torch.rand(1, 3, 32, 32)
    y, cb, cr = rgb_to_ycbcr(rgb)
    target = torch.rand_like(y)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    output = encoder(y, cb, cr)
    (output["carrier"] - target).square().mean().backward()
    assert encoder.head.weight.grad.abs().sum() > 0
    assert encoder.stem.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = encoder(y, cb, cr)
    (output["carrier"] - target).square().mean().backward()
    assert encoder.stem.weight.grad.abs().sum() > 0


def test_ew_host_condition_is_detached_but_carrier_main_path_is_not():
    encoder = WatermarkEncoder()
    message = torch.randint(0, 2, (1, 64)).float()
    host = torch.rand(1, 1, 256, 256, requires_grad=True)
    encoder(host, message)["candidate"].square().mean().backward()
    assert host.grad is None or not bool(torch.count_nonzero(host.grad))

    host = torch.rand(1, 1, 256, 256, requires_grad=True)
    encoder(host, message)["carrier"].sum().backward()
    torch.testing.assert_close(host.grad, torch.ones_like(host))


def test_v2_encoder_and_decoder_intermediate_shapes_for_batch_two():
    encoder, decoder = WatermarkEncoder(), WatermarkDecoder()
    host = torch.rand(2, 1, 256, 256)
    message = torch.randint(0, 2, (2, 64)).float()
    with torch.no_grad():
        water = encoder(host, message)
        assert water["coefficients"].shape == (2, 9, 32, 32)
        assert water["dct_residual"].shape == (2, 64, 32, 32)
        assert water["candidate"].shape == (2, 1, 256, 256)
        assert water["residual"].shape == (2, 1, 256, 256)
        assert decoder(water["carrier"]).shape == (2, 64)


def test_full_loss_backward_reaches_all_four_networks():
    torch.manual_seed(31)
    model = v2_system()
    message = torch.randint(0, 2, (1, 64)).float()
    losses = CleanLoss(DEFAULT_CONFIG["loss"])(
        model(torch.rand(1, 3, 256, 256), message), message
    )
    losses["total"].backward()
    for layer in (model.color_encoder.head, model.watermark_encoder.head,
                  model.color_decoder.chroma_head, model.watermark_decoder.head):
        assert layer.weight.grad is not None
        assert bool(torch.isfinite(layer.weight.grad).all())
        assert layer.weight.grad.abs().sum() > 0


def test_decoder_has_raw_logits_not_sigmoid():
    decoder = WatermarkDecoder()
    with torch.no_grad():
        decoder.head.weight.zero_()
        decoder.head.bias.fill_(-3)
    assert torch.equal(decoder(torch.rand(1, 1, 256, 256)), torch.full((1, 64), -3.0))


@pytest.mark.parametrize("message", [torch.zeros(1, 63), torch.full((1, 64), 0.5)])
def test_bad_message_rejected(message):
    with pytest.raises(ValueError):
        WatermarkEncoder()(torch.rand(1, 1, 256, 256), message)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 256, 248)])
def test_full_ew_and_dw_reject_non_256_inputs(shape):
    with pytest.raises(ValueError, match="256"):
        WatermarkEncoder()(torch.rand(shape), torch.zeros(1, 64))
    with pytest.raises(ValueError, match="256"):
        WatermarkDecoder()(torch.rand(shape))


def test_dc_returns_existing_interface_keys():
    with torch.no_grad():
        output = ColorDecoder()(torch.rand(1, 1, 32, 32))
    assert set(output) == {"rgb", "y", "cb", "cr", "raw_luma_delta",
                           "luma_delta", "z", "zc"}
    assert output["rgb"].shape == (1, 3, 32, 32)


def test_clean_algebra_separates_nonzero_payloads():
    dct = BlockDCT()
    y = torch.rand(2, 1, 16, 16)
    dc, dw = dct.project(torch.randn_like(y), "c"), dct.project(torch.randn_like(y), "w")
    x, s = y + dc + dw, y + dc
    torch.testing.assert_close(x - dct.project(x, "w"), s - dct.project(s, "w"),
                               atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(dct.watermark(x), dct.watermark(y + dw), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(s - dct.project(s, "c"), y - dct.project(y, "c"),
                               atol=2e-6, rtol=2e-6)
