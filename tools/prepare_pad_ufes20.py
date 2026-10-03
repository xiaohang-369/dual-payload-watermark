#!/usr/bin/env python3
"""Frozen PAD-UFES-20 preprocessing v1 and patient split. Never trains a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version

if __package__:
    from .audit_pad_ufes20 import read_metadata, table
    from .pad_patient_split import (DIAGNOSTICS, PATIENT_TARGETS, RATIOS,
                                    SEED, SPLITS, build_patient_vectors, split_patients)
else:
    from audit_pad_ufes20 import read_metadata, table
    from pad_patient_split import (DIAGNOSTICS, PATIENT_TARGETS, RATIOS,
                                   SEED, SPLITS, build_patient_vectors, split_patients)

LINEAGE_FIELDS = ("img_id", "patient_id", "lesion_id", "diagnostic", "lesion_key",
                  "source_path", "prepared_path", "source_width", "source_height",
                  "source_mode", "resize_method", "was_upsampled", "aspect_ratio",
                  "source_sha256", "prepared_sha256")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(root, exclude=()):
    """Hash every original regular file, including archive and earlier reports."""
    result = {}

    def fail(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        folder = Path(directory)
        dirs[:] = sorted(d for d in dirs if folder / d not in exclude)
        for name in [*dirs, *files]:
            if (folder / name).is_symlink():
                raise ValueError(f"Source snapshot refuses symlink: {folder / name}")
        for name in sorted(files):
            path = folder / name
            if not path.is_file():
                raise ValueError(f"Non-regular source file: {path}")
            result[path.relative_to(root).as_posix()] = sha256(path)
    return dict(sorted(result.items()))


def load_source(source):
    data = Path(source).read_bytes()
    with Image.open(io.BytesIO(data)) as check:
        if check.format != "PNG":
            raise ValueError(f"Source must be PNG: {source}")
        check.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"Unsupported source mode {image.mode}: {source}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Animated PNG is not a single original image: {source}")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"RGBA alpha must be entirely 255: {source}")
        width, height = image.size
        properties = {"source_width": width, "source_height": height, "source_mode": image.mode,
                      "resize_method": ("NONE" if image.size == (256, 256) else
                                        "BICUBIC" if width < 256 and height < 256 else "LANCZOS"),
                      "was_upsampled": width < 256 or height < 256, "aspect_ratio": width / height,
                      "source_sha256": hashlib.sha256(data).hexdigest()}
        # Rebuild from channel bytes so no EXIF/ICC metadata is implicitly applied or copied.
        rgb = Image.merge("RGB", image.split()[:3]) if image.mode == "RGBA" else image.copy()
        rgb.info.clear()
    return rgb, properties


def preprocess_image(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Output already exists: {destination}")
    image, properties = load_source(source)
    try:
        if properties["resize_method"] != "NONE":
            resized = image.resize((256, 256), resample=getattr(Image.Resampling, properties["resize_method"]))
            image.close()
            image = resized
        with destination.open("xb") as stream:
            image.save(stream, format="PNG", optimize=False, compress_level=6)
    finally:
        image.close()
    properties["prepared_sha256"] = sha256(destination)
    return properties


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_outputs(output, rows, patient_targets=PATIENT_TARGETS):
    """Read back on-disk artifacts; require exact coverage, metadata and image bytes."""
    output = Path(output).resolve()
    expected = {r["img_id"]: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate expected images")
    seen_images, seen_patients, seen_lesions = set(), set(), set()
    for split, quota in zip(SPLITS, patient_targets):
        manifest = output / "manifests" / f"{split}.json"
        samples = json.loads(manifest.read_text())["samples"]
        patients, lesions, labels = set(), set(), set()
        for row in samples:
            image_id = row["img_id"]
            if image_id in seen_images or image_id not in expected:
                raise ValueError(f"Duplicated or unexpected image: {image_id}")
            if any(row.get(field) != expected[image_id][field]
                   for field in ("patient_id", "lesion_id", "diagnostic")):
                raise ValueError(f"Manifest metadata mismatch: {image_id}")
            relative = Path(row["path"])
            if relative.is_absolute() or relative.as_posix() != f"../images/{image_id}":
                raise ValueError(f"Nonportable or unexpected manifest path: {relative}")
            path = (manifest.parent / relative).resolve()
            if path != output / "images" / image_id:
                raise ValueError(f"Image path escapes prepared root: {relative}")
            with Image.open(path) as img:
                img.load()
                if img.format != "PNG" or img.mode != "RGB" or img.size != (256, 256):
                    raise ValueError(f"Invalid prepared image: {path}")
            seen_images.add(image_id)
            patients.add(row["patient_id"])
            lesions.add((row["patient_id"], row["lesion_id"]))
            labels.add(row["diagnostic"])
        if patients & seen_patients or lesions & seen_lesions:
            raise ValueError("Patient or lesion overlaps between splits")
        if len(patients) != quota:
            raise ValueError(f"Wrong patient count for {split}: {len(patients)}")
        if labels != set(DIAGNOSTICS):
            raise ValueError(f"Missing diagnostic/MEL in {split}")
        seen_patients.update(patients)
        seen_lesions.update(lesions)
    if seen_images != set(expected) or seen_patients != {r["patient_id"] for r in rows}:
        raise ValueError("Image/patient coverage is incomplete")
    if seen_lesions != {(r["patient_id"], r["lesion_id"]) for r in rows}:
        raise ValueError("Lesion coverage is incomplete")
    actual_files = list((output / "images").iterdir())
    if {p.name for p in actual_files} != set(expected) or any(not p.is_file() or p.is_symlink() for p in actual_files):
        raise ValueError("Prepared file inventory mismatch")
    with (output / "preprocessing_manifest.csv").open(newline="") as stream:
        lineage = list(csv.DictReader(stream))
    if len(lineage) != len(expected) or {r["img_id"] for r in lineage} != set(expected):
        raise ValueError("Preprocessing manifest coverage mismatch")
    for row in lineage:
        image_id = row["img_id"]
        if row["prepared_path"] != f"images/{image_id}" or sha256(output / row["prepared_path"]) != row["prepared_sha256"]:
            raise ValueError(f"Prepared hash/path mismatch: {image_id}")
        if any(row[f] != expected[image_id][f] for f in ("patient_id", "lesion_id", "diagnostic")):
            raise ValueError(f"Lineage metadata mismatch: {image_id}")
        if json.loads(row["lesion_key"]) != [row["patient_id"], row["lesion_id"]]:
            raise ValueError(f"Lineage lesion key mismatch: {image_id}")
    return dict.fromkeys(("patient_disjoint", "img_id_disjoint", "lesion_key_disjoint",
                          "all_images_once", "all_patients_once", "exact_patient_quotas",
                          "six_diagnostics_each_split", "mel_patient_and_image_present",
                          "all_prepared_rgb_256_png", "exact_prepared_file_count",
                          "portable_relative_paths", "prepared_hashes_verified"), True)


def describe_split(rows):
    patients, vectors = build_patient_vectors(rows)
    total = vectors.sum(axis=0)
    return {"patient_count": len(patients), "image_count": len(rows), "lesion_count": int(total[1]),
            "diagnostics": {d: {"image_count": int(total[2 + i]), "lesion_count": int(total[8 + i]),
                                "patient_count": int(total[14 + i])} for i, d in enumerate(DIAGNOSTICS)}}


def make_summary(rows, assignment, lineage, evidence, patient_targets, ratios):
    global_counts = describe_split(rows)
    summary = {"version": "prepared_v1", "seed": SEED, "ratios": dict(zip(SPLITS, ratios)),
               "patient_targets": dict(zip(SPLITS, patient_targets)), "diagnostic_order": list(DIAGNOSTICS),
               "environment": {"python": platform.python_version(), "Pillow": pillow_version, "numpy": np.__version__},
               "preprocessing": {"version": 1, "output": "8-bit RGB PNG 256x256",
                                 "alpha": "must be all 255 before dropping A", "was_upsampled": "either source dimension <256",
                                 "resize_counts": dict(Counter(r["resize_method"] for r in lineage)),
                                 "source_modes": dict(Counter(r["source_mode"] for r in lineage)),
                                 "no_other_pixel_transforms": True},
               "optimization": evidence, "global": global_counts, "splits": {}, "deviations": [],
               "geometry": {"short_side_lt256": [], "aspect_ratio_outside_0_9_to_1_1": []}}
    for split, ratio in zip(SPLITS, ratios):
        counts = describe_split([r for r in rows if assignment[r["patient_id"]] == split])
        summary["splits"][split] = counts
        pairs = [(name, counts[name], global_counts[name]) for name in ("patient_count", "image_count", "lesion_count")]
        pairs.extend((f"{d}_{name}", counts["diagnostics"][d][name], global_counts["diagnostics"][d][name])
                     for d in DIAGNOSTICS for name in ("image_count", "lesion_count", "patient_count"))
        for name, actual, overall in pairs:
            target = overall * ratio
            difference = actual - target
            summary["deviations"].append({"split": split, "metric": name, "actual": actual,
                                           "target": target, "signed_difference": difference,
                                           "absolute_difference": abs(difference),
                                           "relative_difference": difference / target if target else 0,
                                           "absolute_relative_difference": abs(difference) / target if target else 0})
    for row in lineage:
        entry = {f: row[f] for f in ("img_id", "patient_id", "lesion_id", "diagnostic",
                                    "source_width", "source_height", "aspect_ratio")}
        entry["split"] = assignment[row["patient_id"]]
        if min(row["source_width"], row["source_height"]) < 256:
            summary["geometry"]["short_side_lt256"].append(entry)
        if row["aspect_ratio"] < .9 or row["aspect_ratio"] > 1.1:
            summary["geometry"]["aspect_ratio_outside_0_9_to_1_1"].append(entry)
    return summary


def report_markdown(summary):
    parts = ["# PAD-UFES-20 prepared v1 与 patient split", "seed=2026；比例 70/15/15。病灶以 (patient_id, lesion_id) 计数。",
             "## 数量", table([{"split": s, **summary["splits"][s]} for s in SPLITS],
                              ["split", "patient_count", "image_count", "lesion_count"]),
             "## 六类分布", table([{"split": s, "diagnostic": d, **summary["splits"][s]["diagnostics"][d]}
                                     for s in SPLITS for d in DIAGNOSTICS],
                                    ["split", "diagnostic", "image_count", "lesion_count", "patient_count"]),
             "## 与理论比例目标的差异", "relative_difference 为带符号比例 (actual-target)/target；absolute_difference 为计数差的绝对值。",
             table(summary["deviations"], ["split", "metric", "actual", "target", "absolute_difference", "relative_difference"]),
             "## 确定性优化", "```json\n" + json.dumps(summary["optimization"], ensure_ascii=False, indent=2) + "\n```",
             "## 冻结 preprocessing v1", "RGB 原通道；RGBA 先验证 alpha 全 255，再去 A。原尺寸为 256×256 不 resize；两边均<256 用 BICUBIC，其余 LANCZOS。",
             "不 crop/padding/EXIF 自动旋转/ICC 颜色变换/线性化/normalization/augmentation；输出不携带源 EXIF/ICC。",
             "```json\n" + json.dumps(summary["preprocessing"], ensure_ascii=False, indent=2) + "\n```"]
    for label, entries in summary["geometry"].items():
        parts += [f"## {label}：{len(entries)} 张", table(entries, ["img_id", "patient_id", "split", "source_width", "source_height", "aspect_ratio"])]
    parts += ["## 自动验收", table([{"check": k, "passed": v} for k, v in summary["validation"].items()], ["check", "passed"]),
              f"原始文件 SHA-256 全部一致：{summary['source_integrity']['compared_files']} 个文件。完整哈希在 source_integrity.json。",
              "## 环境", json.dumps(summary["environment"], ensure_ascii=False),
              "本次仅生成数据和 manifests，未训练模型、未决定 batch/lr/epoch。合成测试与数据准备不代表研究效果验证。"]
    return "\n\n".join(parts) + "\n"


def prepare_dataset(dataset_root, metadata, image_dir, output_dir, *, expected_images=2298,
                    patient_targets=PATIENT_TARGETS, ratios=RATIOS):
    root, metadata, image_dir, output = (Path(p).resolve() for p in (dataset_root, metadata, image_dir, output_dir))
    if output.exists() or Path(output_dir).is_symlink():
        raise ValueError(f"Output already exists: {output}")
    if not root.is_dir() or not metadata.is_file() or not image_dir.is_dir():
        raise ValueError("Dataset root, metadata or image directory missing")
    if not metadata.is_relative_to(root) or not image_dir.is_relative_to(root):
        raise ValueError("Input files must be inside dataset root for complete integrity verification")
    if output.is_relative_to(image_dir) or metadata.is_relative_to(output) or image_dir.is_relative_to(output):
        raise ValueError("Output would overlap original inputs")
    _, rows, schema = read_metadata(metadata)
    if any(schema[k] for k in ("missing_required_fields", "duplicate_fields", "malformed_records")):
        raise ValueError(f"Invalid metadata schema; actual fields: {schema}")
    rows = sorted(({f: r[f] for f in ("img_id", "patient_id", "lesion_id", "diagnostic")} for r in rows), key=lambda r: r["img_id"])
    ids, _ = build_patient_vectors(rows)
    if len(rows) != expected_images or len(ids) != sum(patient_targets):
        raise ValueError(f"Expected {expected_images} images/{sum(patient_targets)} patients; got {len(rows)}/{len(ids)}")
    for row in rows:
        name = row["img_id"]
        if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".png"):
            raise ValueError(f"Unsafe or noncanonical img_id: {name}")
    before = snapshot(root)
    sources = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.name in sources:
                raise ValueError(f"Ambiguous source filename: {path.name}")
            sources[path.name] = path
    if set(sources) != {r["img_id"] for r in rows}:
        raise ValueError("Source PNG inventory and metadata do not match exactly")
    print(f"Preflight: {len(rows)} images, {len(ids)} patients", flush=True)
    for row in rows:
        source = sources[row["img_id"]]
        image, properties = load_source(source)
        image.close()
        if properties["source_sha256"] != before[source.relative_to(root).as_posix()]:
            raise ValueError(f"Source changed during preflight: {source}")
    assignment, evidence = split_patients(rows, patient_targets, ratios)
    repeated, repeated_evidence = split_patients(list(reversed(rows)), patient_targets, ratios)
    if assignment != repeated or evidence != repeated_evidence:
        raise ValueError("Split reproducibility verification failed")
    print(f"Split verified; objective {evidence['initial_objective']:.8f} -> {evidence['final_objective']:.8f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name in ("images", "manifests", "reports"):
            (staging / name).mkdir()
        lineage = []
        for number, row in enumerate(rows, 1):
            source = sources[row["img_id"]]
            properties = preprocess_image(source, staging / "images" / row["img_id"])
            if properties["source_sha256"] != before[source.relative_to(root).as_posix()]:
                raise ValueError(f"Source changed during preparation: {source}")
            lineage.append({**row, **properties, "lesion_key": json.dumps([row["patient_id"], row["lesion_id"]]),
                            "source_path": str(source), "prepared_path": f"images/{row['img_id']}"})
            if number % 250 == 0 or number == len(rows):
                print(f"Prepared {number}/{len(rows)}", flush=True)
        with (staging / "preprocessing_manifest.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=LINEAGE_FIELDS)
            writer.writeheader()
            writer.writerows(lineage)
        for split in SPLITS:
            samples = [{**row, "path": f"../images/{row['img_id']}"} for row in rows if assignment[row["patient_id"]] == split]
            write_json(staging / "manifests" / f"{split}.json", {"samples": samples})
        validation = validate_outputs(staging, rows, patient_targets)
        after = snapshot(root, exclude=(staging,))
        if after != before:
            raise ValueError("Original dataset SHA-256 or file inventory changed")
        integrity = {"algorithm": "SHA-256", "compared_files": len(before), "all_unchanged": True,
                     "scope": str(root), "excluded": "only this newly generated prepared output",
                     "before": before, "after": after}
        write_json(staging / "reports/source_integrity.json", integrity)
        summary = make_summary(rows, assignment, lineage, evidence, patient_targets, ratios)
        summary["validation"] = {**validation, "split_determinism_verified": True, "original_dataset_sha256_unchanged": True}
        summary["source_integrity"] = {k: v for k, v in integrity.items() if k not in ("before", "after")}
        write_json(staging / "reports/split_summary.json", summary)
        (staging / "reports/split_report.md").write_text(report_markdown(summary), encoding="utf-8")
        if output.exists():
            raise ValueError(f"Output appeared during preparation: {output}")
        staging.rename(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"Complete: {output}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "metadata", "image-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        prepare_dataset(args.dataset_root, args.metadata, args.image_dir, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
