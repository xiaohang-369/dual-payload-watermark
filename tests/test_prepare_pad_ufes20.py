import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from tools.pad_patient_split import DIAGNOSTICS, build_patient_vectors, split_patients
from tools.prepare_pad_ufes20 import prepare_dataset, preprocess_image, validate_outputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode,size,method", [
    ("RGB", (256, 256), "NONE"), ("RGBA", (256, 256), "NONE"),
    ("RGB", (80, 120), "BICUBIC"), ("RGB", (400, 500), "LANCZOS"),
    ("RGB", (128, 512), "LANCZOS"), ("RGB", (255, 256), "LANCZOS"),
])
def test_frozen_preprocessing(tmp_path, monkeypatch, mode, size, method):
    source, output = tmp_path / "source.png", tmp_path / "prepared.png"
    values = np.random.default_rng(9).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    rgb = Image.fromarray(values)
    image = rgb.copy()
    if mode == "RGBA":
        image.putalpha(255)
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif, icc_profile=b"synthetic-icc-do-not-transform")
    before = sha(source)
    resize_calls = []
    original = Image.Image.resize

    def record_resize(self, target, resample=None, **kwargs):
        resize_calls.append(resample)
        return original(self, target, resample=resample, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", record_resize)
    record = preprocess_image(source, output)
    assert record["resize_method"] == method
    assert record["was_upsampled"] == (min(size) < 256)
    assert record["source_sha256"] == before
    assert record["prepared_sha256"] == sha(output)
    assert sha(source) == before
    with Image.open(output) as prepared:
        assert prepared.mode == "RGB" and prepared.size == (256, 256)
        assert "icc_profile" not in prepared.info and "exif" not in prepared.info
        if method == "NONE":
            assert resize_calls == []
            np.testing.assert_array_equal(np.array(prepared), values)
        else:
            assert resize_calls == [getattr(Image.Resampling, method)]
            expected = original(rgb, (256, 256), resample=getattr(Image.Resampling, method))
            np.testing.assert_array_equal(np.array(prepared), np.array(expected))


@pytest.mark.parametrize("alpha", [0, 128, 254])
def test_transparency_is_fatal(tmp_path, alpha):
    source, output = tmp_path / "a.png", tmp_path / "output.png"
    image = Image.new("RGBA", (16, 16), (1, 2, 3, 255))
    image.putpixel((0, 0), (1, 2, 3, alpha))
    image.save(source)
    before = sha(source)
    with pytest.raises(ValueError, match="alpha"):
        preprocess_image(source, output)
    assert not output.exists() and sha(source) == before


@pytest.mark.parametrize("mode", ["L", "P", "LA"])
def test_other_modes_rejected(tmp_path, mode):
    source = tmp_path / "bad.png"
    Image.new(mode, (16, 16)).save(source)
    with pytest.raises(ValueError, match="mode"):
        preprocess_image(source, tmp_path / "out.png")


def records():
    rows = []
    for patient in range(16):
        for diagnostic in DIAGNOSTICS:
            count = 2 if diagnostic == "MEL" and patient % 3 == 0 else 1
            for i in range(count):
                rows.append({"patient_id": f"p{patient:02}", "lesion_id": diagnostic,
                             "diagnostic": diagnostic, "img_id": f"p{patient:02}_{diagnostic}_{i}.png"})
    return rows


def test_patient_vectors_and_deterministic_multilabel_split():
    rows = records()
    ids, vectors = build_patient_vectors(rows)
    assert vectors.shape == (16, 20)
    assert vectors[0].tolist() == [7, 6, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    first, evidence = split_patients(rows, (8, 4, 4), (.5, .25, .25))
    second, second_evidence = split_patients(list(reversed(rows)), (8, 4, 4), (.5, .25, .25))
    assert first == second and evidence == second_evidence
    assert [sum(s == split for s in first.values()) for split in ("train", "val", "test")] == [8, 4, 4]
    assert evidence["final_objective"] <= evidence["initial_objective"]
    for split in ("train", "val", "test"):
        selected = [r for r in rows if first[r["patient_id"]] == split]
        assert {r["diagnostic"] for r in selected} == set(DIAGNOSTICS)
        assert any(r["diagnostic"] == "MEL" for r in selected)
    assert len({r["img_id"] for r in rows}) == len(rows)


def source_dataset(tmp_path):
    root = tmp_path / "dataset"
    images = root / "raw/images"
    images.mkdir(parents=True)
    rows = records()
    metadata = root / "raw/metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        Image.new("RGB", (16, 16), (1, 2, 3)).save(images / row["img_id"])
    (root / "old-audit.txt").write_text("must remain unchanged")
    return root, metadata, images, rows


def test_generation_integrity_and_portable_manifests(tmp_path):
    root, metadata, images, rows = source_dataset(tmp_path)
    before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
    out = root / "prepared_v1"
    summary = prepare_dataset(root, metadata, images, out, expected_images=len(rows),
                              patient_targets=(8, 4, 4), ratios=(.5, .25, .25))
    assert all(summary["validation"].values())
    assert len(list((out / "images").glob("*.png"))) == len(rows)
    assert {name: sha(root / name) for name in before} == before
    with (out / "preprocessing_manifest.csv").open() as stream:
        lineage = list(csv.DictReader(stream))
    assert len(lineage) == len(rows)
    assert all(not Path(r["prepared_path"]).is_absolute() for r in lineage)
    seen = []
    for split in ("train", "val", "test"):
        manifest = out / "manifests" / f"{split}.json"
        entries = json.loads(manifest.read_text())["samples"]
        for row in entries:
            assert not Path(row["path"]).is_absolute()
            assert (manifest.parent / row["path"]).is_file()
            seen.append(row["img_id"])
    assert len(seen) == len(set(seen)) == len(rows)
    relocated = tmp_path / "relocated"
    shutil.move(out, relocated)
    assert all(validate_outputs(relocated, rows, (8, 4, 4)).values())
    from dual_payload.data import ManifestDataset
    prepared_data = ManifestDataset(relocated / "manifests/train.json")
    assert prepared_data[0]["rgb"].shape == (3, 256, 256)
    manifest = relocated / "manifests/train.json"
    data = json.loads(manifest.read_text())
    data["samples"].append(data["samples"][0])
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate_outputs(relocated, rows, (8, 4, 4))


def test_preflight_alpha_failure_and_existing_output_not_touched(tmp_path):
    root, metadata, images, rows = source_dataset(tmp_path)
    target = root / "prepared_v1"
    bad = images / rows[-1]["img_id"]
    Image.new("RGBA", (16, 16), (0, 0, 0, 254)).save(bad)
    with pytest.raises(ValueError, match="alpha"):
        prepare_dataset(root, metadata, images, target, expected_images=len(rows),
                        patient_targets=(8, 4, 4), ratios=(.5, .25, .25))
    assert not target.exists()
    target.mkdir()
    keep = target / "keep.txt"
    keep.write_text("existing")
    with pytest.raises(ValueError, match="exist"):
        prepare_dataset(root, metadata, images, target, expected_images=len(rows),
                        patient_targets=(8, 4, 4), ratios=(.5, .25, .25))
    assert keep.read_text() == "existing"


def test_insufficient_mel_rejected():
    rows = [r for r in records() if r["diagnostic"] != "MEL" or r["patient_id"] == "p00"]
    with pytest.raises(ValueError, match="MEL"):
        split_patients(rows, (8, 4, 4), (.5, .25, .25))


def test_source_change_aborts_publication(tmp_path, monkeypatch):
    import tools.prepare_pad_ufes20 as preparation

    root, metadata, images, rows = source_dataset(tmp_path)
    original = preparation.preprocess_image
    changed = False

    def simulate_concurrent_change(*args):
        nonlocal changed
        value = original(*args)
        if not changed:
            (root / "old-audit.txt").write_text("external concurrent modification")
            changed = True
        return value

    monkeypatch.setattr(preparation, "preprocess_image", simulate_concurrent_change)
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_dataset(root, metadata, images, root / "prepared_v1", expected_images=len(rows),
                        patient_targets=(8, 4, 4), ratios=(.5, .25, .25))
    assert not (root / "prepared_v1").exists()
    assert not list(root.glob(".prepared_v1-*"))


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "lesion_conflict"])
def test_invalid_metadata_rejected(mutation):
    rows = records()
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unknown":
        rows[0]["diagnostic"] = "unexpected"
    else:
        rows.append({**rows[0], "img_id": "conflict.png", "diagnostic": "BCC"})
    with pytest.raises(ValueError):
        split_patients(rows, (8, 4, 4), (.5, .25, .25))
