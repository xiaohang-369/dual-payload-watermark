"""Image metrics shared by medical file evaluation."""
import torch

from dual_payload.metrics import psnr, psnr_per_sample, ssim, ssim_per_sample


def test_metrics_identity():
    image = torch.rand(2, 3, 16, 16)
    assert float(psnr(image, image)) == 120
    assert psnr_per_sample(image, image).tolist() == [120, 120]
    torch.testing.assert_close(ssim(image, image), torch.tensor(1.))
    torch.testing.assert_close(ssim_per_sample(image, image), torch.ones(2))


def test_psnr_reports_each_image_before_averaging():
    target = torch.zeros(2, 3, 16, 16)
    prediction = torch.stack((torch.full_like(target[0], 0.1), torch.full_like(target[0], 0.01)))
    torch.testing.assert_close(psnr_per_sample(prediction, target), torch.tensor([20., 40.]))
    torch.testing.assert_close(psnr(prediction, target), torch.tensor(30.))
    scores = ssim_per_sample(prediction, target)
    assert torch.isfinite(scores).all() and (scores < 1).all()
    torch.testing.assert_close(ssim(prediction, target), scores.mean())
