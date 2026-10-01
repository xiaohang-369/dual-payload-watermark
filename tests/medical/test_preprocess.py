"""Source-format normalization checks; no claim of medical training performance."""
import hashlib

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms, ImageOps

from dual_payload.medical.data import MedicalDataset, create_manifest
from dual_payload.medical.preprocess import integer_pixels, prepare_work_image
from dual_payload.medical.profile import FIXED, read_json, template


def linear_rgb_icc():
    """An sRGB-primary RGB profile with linear TRCs, built entirely in the test."""
    raw = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    offset = len(raw)
    # ICC parametric curve type 0: Y = X**1, encoded as s15Fixed16 gamma.
    curve = b'para' + bytes(8) + (65536).to_bytes(4, 'big')
    raw.extend(curve)
    raw[:4] = len(raw).to_bytes(4, 'big')
    raw[84:100] = bytes(16)  # No stale profile ID after changing the curves.
    for i in range(int.from_bytes(raw[128:132], 'big')):
        pos = 132 + 12*i
        if raw[pos:pos+4] in (b'rTRC', b'gTRC', b'bTRC'):
            raw[pos+4:pos+8] = offset.to_bytes(4, 'big')
            raw[pos+8:pos+12] = len(curve).to_bytes(4, 'big')
    return bytes(raw)


@pytest.mark.parametrize('orientation', [1, 6])
def test_opaque_rgba_matches_original_rgb_and_keeps_source_bytes(tmp_path, orientation):
    pixels = np.random.default_rng(7).integers(0, 256, (37, 59, 3), dtype=np.uint8)
    rgb = Image.fromarray(pixels)
    exif = Image.Exif(); exif[274] = orientation
    source = tmp_path/'rgb.png'; opaque = tmp_path/'rgba.png'
    rgb.save(source, exif=exif)
    rgb.convert('RGBA').save(opaque, exif=exif)
    before = opaque.read_bytes()
    expected, rect = prepare_work_image(source)
    actual, actual_rect = prepare_work_image(opaque)
    assert rect == actual_rect
    assert torch.equal(actual, expected)
    assert opaque.read_bytes() == before
    # Preserve the pre-existing RGB geometry and pixel processing exactly.
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        x, y, w, h = rect
        resized = np.asarray(im.resize((w, h), Image.Resampling.BICUBIC))
        padded = np.pad(resized, ((y, 256-h-y), (x, 256-w-x), (0, 0)), mode='edge')
    assert np.array_equal(integer_pixels(actual)[0].transpose(1, 2, 0), padded)


@pytest.mark.parametrize('mode', ['RGB', 'RGBA'])
def test_embedded_linear_rgb_icc_is_converted_not_stripped(tmp_path, mode):
    path = tmp_path/'tagged.png'
    Image.new(mode, (256, 256), (128, 128, 128) if mode == 'RGB' else (128, 128, 128, 255)).save(
        path, icc_profile=linear_rgb_icc())
    before = path.read_bytes()
    work, rect = prepare_work_image(path)
    # Linear 128/255 becomes approximately 188/255 after sRGB encoding.
    assert rect == (0, 0, 256, 256)
    assert np.abs(integer_pixels(work).astype(int) - 188).max() <= 1
    assert path.read_bytes() == before


@pytest.mark.parametrize('alpha', [0, 127, 254])
def test_one_nonopaque_pixel_is_rejected(tmp_path, alpha):
    path = tmp_path/'transparent.png'
    im = Image.new('RGBA', (20, 20), (10, 20, 30, 255))
    im.putpixel((7, 9), (10, 20, 30, alpha)); im.save(path)
    with pytest.raises(ValueError, match='transparency'):
        prepare_work_image(path)


@pytest.mark.parametrize('icc', [b'broken profile', ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB')).tobytes()])
def test_invalid_or_non_rgb_icc_is_rejected(tmp_path, icc):
    path = tmp_path/'bad-icc.png'
    Image.new('RGB', (20, 20)).save(path, icc_profile=icc)
    with pytest.raises(ValueError, match='ICC'):
        prepare_work_image(path)


def test_rgb_transparency_and_multiple_frames_are_still_rejected(tmp_path):
    path = tmp_path/'transparent.png'
    Image.new('RGB', (20, 20), (1, 2, 3)).save(path, transparency=(1, 2, 3))
    with pytest.raises(ValueError, match='transparency'):
        prepare_work_image(path)
    animated = tmp_path/'animated.png'
    Image.new('RGB', (20, 20), 'red').save(animated, save_all=True,
        append_images=[Image.new('RGB', (20, 20), 'blue')], duration=100)
    with pytest.raises(ValueError, match='single RGB8'):
        prepare_work_image(animated)


def test_manifest_and_dataset_share_normalization_and_preserve_source_hashes(tmp_path):
    root = tmp_path/'data'; root.mkdir()
    rows = ['patient_id,lesion_id,img_id,diagnostic']
    for i, mode in enumerate(('RGB', 'RGBA', 'RGB')):
        path = root/f'{i}.png'
        color = (20 + 50*i, 80, 120) + ((255,) if mode == 'RGBA' else ())
        Image.new(mode, (32, 48), color).save(path, **({'icc_profile': linear_rgb_icc()} if i == 2 else {}))
        rows.append(f'p{i},l{i},{i}.png,synthetic')
    metadata = root/'metadata.csv'; metadata.write_text('\n'.join(rows) + '\n')
    path = tmp_path/'manifest.json'
    manifest = create_manifest(root, metadata, path, (0.7, 0.15, 0.15), 2026)
    assert manifest['preprocess'] == FIXED['preprocess']
    for split in ('train', 'valid', 'test'):
        dataset = MedicalDataset(path, {'pad': str(root)}, split)
        sample = dataset[0]; record = dataset.records[0]
        source = root/record['path']
        expected, rect = prepare_work_image(source)
        assert torch.equal(sample['rgb'], expected[0])
        assert sample['content_rect'].tolist() == list(rect)
        assert record['file_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    from pathlib import Path
    assert read_json(Path(__file__).resolve().parents[2]/'configs'/'medical_v1.template.json') == template()
