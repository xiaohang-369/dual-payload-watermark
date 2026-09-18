import pytest
import torch

from dual_payload.transforms import BlockDCT, WATERMARK_COORDS, rgb_to_ycbcr, rms_cap, ycbcr_to_rgb


def test_colour_roundtrip_and_zero_centered_chroma():
    rgb = torch.rand(2, 3, 16, 24)
    y, cb, cr = rgb_to_ycbcr(rgb)
    torch.testing.assert_close(ycbcr_to_rgb(y, cb, cr), rgb, atol=3e-7, rtol=1e-6)
    assert cb.abs().max() <= 0.5 and cr.abs().max() <= 0.5
    grey = torch.full((1, 3, 8, 8), 0.5)
    _, cb, cr = rgb_to_ycbcr(grey)
    torch.testing.assert_close(cb, torch.zeros_like(cb), atol=1e-7, rtol=0)
    torch.testing.assert_close(cr, torch.zeros_like(cr), atol=1e-7, rtol=0)


def test_dct_roundtrip_and_energy():
    dct = BlockDCT()
    image = torch.randn(2, 1, 16, 24, requires_grad=True)
    coefficients = dct(image)
    assert coefficients.shape == (2, 64, 2, 3)
    torch.testing.assert_close(dct.inverse(coefficients), image, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(coefficients.square().sum(), image.square().sum(), rtol=2e-6, atol=1e-5)
    dct.inverse(coefficients).sum().backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image), atol=2e-6, rtol=2e-6)
    assert not list(dct.parameters())


def test_band_completeness_orthogonality_and_idempotence():
    dct = BlockDCT()
    assert [int(getattr(dct, 'mask_' + name).sum()) for name in ('0', 'w', 'c')] == [16, 9, 39]
    image = torch.randn(2, 1, 16, 16)
    bands = {name: dct.project(image, name) for name in ('0', 'w', 'c')}
    torch.testing.assert_close(sum(bands.values()), image, atol=2e-6, rtol=2e-6)
    for first, value in bands.items():
        torch.testing.assert_close(dct.project(value, first), value, atol=1e-6, rtol=2e-6)
        for second in bands:
            if first != second:
                assert float(dct.project(value, second).abs().max()) < 1e-6


def test_watermark_frontend_has_exact_coordinate_order():
    dct = BlockDCT()
    coefficients = torch.zeros(1, 64, 1, 1)
    for number, (u, v) in enumerate(WATERMARK_COORDS, start=1):
        coefficients[0, u * 8 + v] = number
    selected = dct.watermark(dct.inverse(coefficients)).flatten()
    torch.testing.assert_close(selected, torch.arange(1, 10).float(), atol=3e-6, rtol=1e-6)


def test_rms_cap_is_per_image_and_preserves_band():
    dct = BlockDCT()
    value = dct.project(torch.randn(2, 1, 16, 16), 'w')
    value[0] *= 100
    cap = rms_cap(value, 2 / 255)
    assert bool((cap.square().flatten(1).mean(1).sqrt() <= 2 / 255 + 1e-8).all())
    for index in range(2):
        torch.testing.assert_close(cap[index:index + 1], rms_cap(value[index:index + 1], 2 / 255))
    assert float(dct.project(cap, 'c').abs().max()) < 1e-8
    zeros = torch.zeros(1, 1, 8, 8, requires_grad=True)
    rms_cap(zeros, 2 / 255).sum().backward()
    assert bool(torch.isfinite(zeros.grad).all())


@pytest.mark.parametrize('shape', [(1, 1, 15, 16), (1, 3, 16, 16), (1, 1, 0, 8)])
def test_bad_dct_shapes_are_rejected(shape):
    with pytest.raises(ValueError):
        BlockDCT()(torch.zeros(shape))
