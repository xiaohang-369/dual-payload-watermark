import json
import numpy as np
from PIL import Image
import pytest
import torch
from dual_payload.data import ManifestDataset, fixed_message, load_rgb_image, ensure_disjoint
from dual_payload.metrics import psnr, ssim


def test_prepared_image_and_manifest_messages(tmp_path):
    pixels = np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
    Image.fromarray(pixels).save(tmp_path/'work.png')
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'samples':[{'path':'work.png','patient_id':'p1'},
                                         {'path':'work.png','message':[1]*256}]}))
    data=ManifestDataset(path)
    assert torch.equal(data[0]['rgb'],torch.from_numpy(pixels.astype(np.float32)/255).permute(2,0,1))
    assert torch.equal(data[0]['message'],data[0]['message'])
    assert data[1]['message'].tolist()==[1]*256
    train=ManifestDataset(path,training=True)
    assert train[0]['message'].shape==(256,)
    assert not torch.equal(train[0]['message'],train[0]['message'])
    with pytest.raises(ValueError,match='overlap'): ensure_disjoint(data.paths,data.paths)
    with pytest.raises(ValueError): fixed_message(0,1,message_bits=64)


@pytest.mark.parametrize('size,mode',[((257,256),'RGB'),((128,128),'RGB'),((256,256),'L')])
def test_no_automatic_medical_preprocessing(tmp_path,size,mode):
    path=tmp_path/'image.png'; Image.new(mode,size).save(path)
    with pytest.raises(ValueError,match='already'): load_rgb_image(path)


def test_metrics_identity():
    image=torch.rand(2,3,16,16)
    assert float(psnr(image,image))==120
    torch.testing.assert_close(ssim(image,image),torch.tensor(1.))
