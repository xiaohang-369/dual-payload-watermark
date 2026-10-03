#!/usr/bin/env python3
"""Read-only PAD-UFES-20 inventory. No model or preprocessing imports.

Exit codes: 0 = complete without findings, 1 = complete with findings,
2 = invalid inputs/schema or incomplete audit. See tools/README.md for counting
rules. Input identifiers are preserved verbatim; no field or label mapping.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import warnings

from PIL import Image


REQUIRED = ("patient_id", "lesion_id", "img_id", "diagnostic")
EXPECTED = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ALPHA_FIELDS = (
    "alpha_min", "alpha_max", "alpha_unique_count", "non_opaque_pixel_count",
    "fully_transparent_pixel_count", "partially_transparent_pixel_count",
    "total_pixel_count", "non_opaque_ratio",
)
EXIT_CODES = {"complete_clean": 0, "complete_with_findings": 1}
IMAGE_FIELDS = (
    "img_id", "patient_id", "lesion_id", "diagnostic", "path", "width",
    "height", "aspect_ratio", "mode", "file_size", "format", "status",
    "metadata_record_numbers", "error", "lesion_keys", *ALPHA_FIELDS,
)
COUNTING_RULES = [
    "metadata 字段名严格匹配；ID 与 diagnostic 按原字符串统计，不去空格、不改大小写、不猜测映射。",
    "缺失指空值或纯空白；NA、NULL 等非空字符串保留原值。CSV 使用 UTF-8（允许 BOM）。",
    "metadata 总行数为 CSV 数据记录数（不含表头）；唯一 ID 排除缺失值。",
    "image_count 按唯一非空 img_id 计数，含缺文件记录；重复 metadata 不增加 image_count。",
    "lesion_key=(patient_id, lesion_id)；仅两个 ID 均非空时构造，不从 img_id 猜测或补齐。原始 lesion_id 保留原文。",
    "lesion_summary、images_per_lesion 和所有 lesion_count 按唯一 lesion_key 统计；patient_count 按唯一非空 patient_id 计数。",
    "raw lesion_id 跨患者复用是数据标识语义现象，单列观察，不作为数据错误；不修改原始 metadata。",
    "仅同一 lesion_key 内多个非空 diagnostic 构成 lesion diagnostic conflict；缺失 diagnostic 单独报告。",
    "诊断 image_percent 分母为 metadata 全部唯一非空 img_id；冲突 ID 可出现在多类中，比例不保证相加为 100%。",
    "图片统计按实际文件路径各计一次，包含未引用及扩展名异常但可解码的图片，不受重复 metadata 影响。",
    "递归扫描普通目录，不跟随目录符号链接（列为异常）；文件符号链接只读并单列。",
    "img_id 只按完整文件名精确对应，不添加扩展名、不改大小写、不选择重复文件名中的某一张。",
    "image_properties 每个图片文件一行，损坏/读取受限图片保留空属性；未匹配文件的关联字段为空。",
    "同一文件关联多个不同字段值时，该 CSV 单元格为 JSON 数组；不选取或覆盖冲突值。",
    "属性基于磁盘像素，不执行 EXIF 旋转或颜色转换；Pillow verify 后重新打开并完整 load。",
    "RGBA alpha 直接读取 A 通道的 0..255 直方图，不做 RGBA→RGB；非 RGBA 的 alpha 字段为空。",
    "non_opaque_ratio 为 alpha<255 的像素数/总像素数，范围 0..1；其他 percent 字段以百分比表示。",
    "几何异常列表完整列出 ratio<0.9 或 ratio>1.1 以及短边<256 的可读取图片，不执行处理。",
    "分位数采用排序后 (n-1)*p 位置线性插值；无有效样本时为 null，比例为 0。",
    "不生成 split，不决定或执行 resize/crop/padding/augmentation/normalization，不保存处理图片。",
]


def present(value: str) -> bool:
    return bool(value.strip())


def unique(rows: list[dict], field: str) -> list[str]:
    return sorted({row[field] for row in rows if present(row[field])})


def lesion_keys(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted({(row["patient_id"], row["lesion_id"]) for row in rows
                   if present(row["patient_id"]) and present(row["lesion_id"])})


def describe(values, percentiles=(5, 25, 75, 95)) -> dict:
    values = sorted(values)
    result = {"count": len(values), "min": None, "max": None,
              "mean": None, "median": None}
    result.update({f"p{p:02d}": None for p in percentiles})
    if values:
        result.update(min=values[0], max=values[-1],
                      mean=statistics.mean(values), median=statistics.median(values))
        for p in percentiles:
            index = (len(values) - 1) * p / 100
            low, high = math.floor(index), math.ceil(index)
            result[f"p{p:02d}"] = values[low] + (values[high] - values[low]) * (index - low)
    return result


def read_metadata(path: Path) -> tuple[list[str], list[dict], dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        fields = next(reader, [])
        schema = {
            "actual_fields": fields,
            "missing_required_fields": sorted(set(REQUIRED) - set(fields)),
            "duplicate_fields": sorted(k for k, n in Counter(fields).items() if n > 1),
            "malformed_records": [],
        }
        rows = []
        for number, values in enumerate(reader, 1):
            if len(values) != len(fields):
                schema["malformed_records"].append({"record_number": number,
                                                    "field_count": len(values)})
            else:
                rows.append(dict(zip(fields, values), _record_number=number))
    return fields, rows, schema


def metadata_audit(rows: list[dict]) -> tuple[dict, list, list, list, dict, dict]:
    by_patient, by_raw_lesion, by_image = defaultdict(list), defaultdict(list), defaultdict(list)
    by_lesion = defaultdict(list)
    for row in rows:
        for field, groups in (("patient_id", by_patient), ("lesion_id", by_raw_lesion),
                              ("img_id", by_image)):
            if present(row[field]):
                groups[row[field]].append(row)
        if present(row["patient_id"]) and present(row["lesion_id"]):
            by_lesion[row["patient_id"], row["lesion_id"]].append(row)
    patients = [{"patient_id": key, "image_count": len(unique(group, "img_id")),
                 "lesion_count": len(lesion_keys(group)),
                 "diagnostics": unique(group, "diagnostic")}
                for key, group in sorted(by_patient.items())]
    lesions = [{"patient_id": key[0], "lesion_id": key[1], "lesion_key": list(key),
                "image_count": len(unique(group, "img_id")),
                "diagnostic": unique(group, "diagnostic")}
               for key, group in sorted(by_lesion.items())]
    labels = sorted(set(EXPECTED) | {r["diagnostic"] for r in rows if present(r["diagnostic"])})
    if any(not present(r["diagnostic"]) for r in rows):
        labels.append("")
    diagnostics = []
    for label in labels:
        group = [r for r in rows if (r["diagnostic"] == label if label else not present(r["diagnostic"]))]
        count = len(unique(group, "img_id"))
        diagnostics.append({"diagnostic": label, "image_count": count,
                            "image_percent": count / len(by_image) * 100 if by_image else 0,
                            "lesion_count": len(lesion_keys(group)),
                            "patient_count": len(unique(group, "patient_id"))})
    findings = {
        "missing_values": {field: [r["_record_number"] for r in rows if not present(r[field])]
                           for field in REQUIRED},
        "duplicate_img_ids": [{"img_id": key, "record_numbers": [r["_record_number"] for r in group]}
                              for key, group in sorted(by_image.items()) if len(group) > 1],
        "patients_with_multiple_diagnostics": [r for r in patients if len(r["diagnostics"]) > 1],
        "lesion_diagnostic_conflicts": [r for r in lesions if len(r["diagnostic"]) > 1],
        "images_with_conflicting_metadata": [
            {"img_id": key, **{f: unique(group, f) for f in ("patient_id", "lesion_id", "diagnostic")}}
            for key, group in sorted(by_image.items())
            if any(len(unique(group, f)) > 1 for f in ("patient_id", "lesion_id", "diagnostic"))],
        "unknown_diagnostics": sorted({r["diagnostic"] for r in rows
                                       if present(r["diagnostic"]) and r["diagnostic"] not in EXPECTED}),
        "surrounding_whitespace": [{"record_number": r["_record_number"], "field": f, "value": r[f]}
                                   for r in rows for f in REQUIRED if present(r[f]) and r[f] != r[f].strip()],
        "img_id_extension_or_path_anomalies": [key for key in sorted(by_image)
                                               if Path(key).suffix != ".png" or "/" in key or "\\" in key],
    }
    observations = {"reused_lesion_ids_across_patients": [
        {"lesion_id": key, "patient_ids": unique(group, "patient_id")}
        for key, group in sorted(by_raw_lesion.items()) if len(unique(group, "patient_id")) > 1]}
    counts = {"total_rows": len(rows), "unique_patient_ids": len(by_patient),
              "raw_unique_lesion_ids": len(by_raw_lesion),
              "unique_patient_lesion_pairs": len(by_lesion),
              "reused_lesion_ids_across_patients": len(observations["reused_lesion_ids_across_patients"]),
              "lesion_diagnostic_conflict_count": len(findings["lesion_diagnostic_conflicts"]),
              "unique_img_ids": len(by_image),
              "missing_counts": {f: len(v) for f, v in findings["missing_values"].items()}}
    levels = {"images_per_patient": describe([r["image_count"] for r in patients], (90, 95, 99)),
              "lesions_per_patient": describe([r["lesion_count"] for r in patients], (90, 95, 99)),
              "images_per_lesion": describe([r["image_count"] for r in lesions], (90, 95, 99)),
              "top20_patients_by_images": sorted(patients, key=lambda r: (-r["image_count"], r["patient_id"]))[:20],
              "top20_patients_by_lesions": sorted(patients, key=lambda r: (-r["lesion_count"], r["patient_id"]))[:20]}
    return {**counts, "hierarchy": levels}, patients, lesions, diagnostics, findings, observations


def inventory(image_dir: Path) -> tuple[list[Path], dict]:
    files = []
    findings = {"skipped_directory_symlinks": [], "file_symlinks": [], "non_regular_entries": []}

    def fail(error):
        raise error

    for root, directories, names in os.walk(image_dir, followlinks=False, onerror=fail):
        for name in sorted(directories):
            path = Path(root) / name
            if path.is_symlink():
                findings["skipped_directory_symlinks"].append(str(path))
        directories[:] = sorted(n for n in directories if not (Path(root) / n).is_symlink())
        for name in sorted(names):
            path = Path(root) / name
            if path.is_symlink():
                findings["file_symlinks"].append(str(path))
            if path.is_file():
                files.append(path)
            else:
                findings["non_regular_entries"].append(str(path))
    return sorted(files), findings


def alpha_properties(img: Image.Image) -> dict:
    """Inspect the original RGBA alpha values without conversion or compositing."""
    histogram = img.getchannel("A").histogram()
    values = [value for value, count in enumerate(histogram) if count]
    total = img.width * img.height
    non_opaque = total - histogram[255]
    return {"alpha_min": min(values), "alpha_max": max(values),
            "alpha_unique_count": len(values), "non_opaque_pixel_count": non_opaque,
            "fully_transparent_pixel_count": histogram[0],
            "partially_transparent_pixel_count": sum(histogram[1:255]),
            "total_pixel_count": total, "non_opaque_ratio": non_opaque / total}


def inspect_images(files: list[Path], rows: list[dict]) -> tuple[list[dict], dict]:
    refs = defaultdict(list)
    names = defaultdict(list)
    folded = defaultdict(list)
    for row in rows:
        if present(row["img_id"]):
            refs[row["img_id"]].append(row)
    for path in files:
        names[path.name].append(str(path))
        folded[path.name.casefold()].append(str(path))
    findings = {
        "missing_images": [], "ambiguous_image_matches": [], "unreferenced_pngs": [],
        "duplicate_filenames": {k: v for k, v in sorted(names.items()) if len(v) > 1},
        "case_insensitive_filename_collisions": {k: v for k, v in sorted(folded.items()) if len(v) > 1},
        "case_mismatch_candidates": [], "extension_anomalies": [], "format_anomalies": [],
        "decode_errors": [], "pillow_warnings": [], "non_image_files": [],
    }
    for key in sorted(refs):
        if not names[key]:
            findings["missing_images"].append(key)
            if folded[key.casefold()]:
                findings["case_mismatch_candidates"].append({"img_id": key, "paths": folded[key.casefold()]})
        elif len(names[key]) > 1:
            findings["ambiguous_image_matches"].append({"img_id": key, "paths": names[key]})
    properties = []
    for path in files:
        linked = refs.get(path.name, [])
        row = {f: unique(linked, f) for f in REQUIRED}
        row.update(path=str(path), width=None, height=None, aspect_ratio=None,
                   mode="", file_size=None, format="", status="", error="",
                   lesion_keys=[list(key) for key in lesion_keys(linked)],
                   metadata_record_numbers=[r["_record_number"] for r in linked])
        row.update({field: None for field in ALPHA_FIELDS})
        # Unreferenced files still need a usable image identifier in the inventory.
        if not linked:
            row["img_id"] = [path.name]
        suspected_png = path.suffix.lower() == ".png"
        # Filesystem read failures must not be mistaken for unrelated text files.
        # Let the caller mark the audit incomplete if even the header is unreadable.
        row["file_size"] = path.stat().st_size
        with path.open("rb") as stream:
            suspected_png = suspected_png or stream.read(8) == PNG_SIGNATURE
        opened = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with Image.open(path) as img:
                    opened = True
                    row["format"] = img.format
                    img.verify()
                with Image.open(path) as img:
                    img.load()
                    if img.mode == "RGBA":
                        row.update(alpha_properties(img))
                    row.update(width=img.width, height=img.height,
                               aspect_ratio=img.width / img.height, mode=img.mode, status="readable")
            except Exception as error:
                # Pillow plugins may also raise EOFError, struct.error, etc.
                # Isolate each decoder failure while preserving its exact type.
                row.update(status="decode_error", error=f"{type(error).__name__}: {error}")
            for warning in caught:
                findings["pillow_warnings"].append({"path": str(path), "warning": str(warning.message)})
        if not (suspected_png or linked or opened):
            findings["non_image_files"].append({"path": str(path), "detail": row["error"]})
            continue
        properties.append(row)
        if suspected_png and not linked:
            findings["unreferenced_pngs"].append(str(path))
        if path.suffix != ".png":
            findings["extension_anomalies"].append({"path": str(path), "extension": path.suffix})
        if row["format"] and row["format"] != "PNG":
            findings["format_anomalies"].append({"path": str(path), "format": row["format"]})
        if row["status"] != "readable":
            findings["decode_errors"].append({"path": str(path), "error": row["error"]})
    return properties, findings


def alpha_statistics(properties: list[dict]) -> dict:
    rgba = [r for r in properties if r["status"] == "readable" and r["mode"] == "RGBA"]
    transparent = [r for r in rgba if r["non_opaque_pixel_count"] > 0]
    return {
        "audited_rgba_images": len(rgba),
        "fully_opaque_rgba_images": sum(r["non_opaque_pixel_count"] == 0 for r in rgba),
        "rgba_images_with_any_transparency": len(transparent),
        "rgba_images_with_partial_alpha": sum(r["partially_transparent_pixel_count"] > 0 for r in rgba),
        "rgba_images_with_alpha_zero": sum(r["fully_transparent_pixel_count"] > 0 for r in rgba),
        "alpha_min_range": {"min": min((r["alpha_min"] for r in rgba), default=None),
                            "max": max((r["alpha_min"] for r in rgba), default=None)},
        "alpha_max_range": {"min": min((r["alpha_max"] for r in rgba), default=None),
                            "max": max((r["alpha_max"] for r in rgba), default=None)},
        "top20_by_non_opaque_ratio": [
            {f: r[f] for f in ("img_id", "path", *ALPHA_FIELDS)}
            for r in sorted(transparent, key=lambda r: (-r["non_opaque_ratio"], r["img_id"], r["path"]))[:20]],
    }


def geometry_lists(properties: list[dict]) -> dict:
    def entry(row):
        return {**{f: row[f] for f in ("img_id", "path", "width", "height", "aspect_ratio")},
                "short_side": min(row["width"], row["height"])}

    readable = [r for r in properties if r["status"] == "readable"]
    return {
        "aspect_ratio_outside_0_9_to_1_1": [entry(r) for r in readable
                                            if r["aspect_ratio"] < .9 or r["aspect_ratio"] > 1.1],
        "short_side_lt256": [entry(r) for r in readable if min(r["width"], r["height"]) < 256],
    }


def image_statistics(properties: list[dict]) -> dict:
    readable = [r for r in properties if r["status"] == "readable"]
    sizes = Counter((r["width"], r["height"]) for r in readable)
    modes = Counter(r["mode"] for r in readable)
    ratios = [r["aspect_ratio"] for r in readable]
    buckets = [
        ("ratio < 0.5", lambda r: r < 0.5),
        ("0.5 <= ratio < 0.75", lambda r: 0.5 <= r < 0.75),
        ("0.75 <= ratio < 0.9", lambda r: 0.75 <= r < 0.9),
        ("0.9 <= ratio <= 1.1", lambda r: 0.9 <= r <= 1.1),
        ("1.1 < ratio <= 1.33", lambda r: 1.1 < r <= 1.33),
        ("1.33 < ratio <= 2.0", lambda r: 1.33 < r <= 2.0),
        ("ratio > 2.0", lambda r: r > 2.0),
    ]
    return {
        "inventoried_image_files": len(properties), "readable_image_files": len(readable),
        "unreadable_image_files": len(properties) - len(readable),
        **{f: describe([r[f] for r in readable]) for f in ("width", "height", "aspect_ratio", "file_size")},
        "orientation": {"portrait": sum(r["width"] < r["height"] for r in readable),
                        "square": sum(r["width"] == r["height"] for r in readable),
                        "landscape": sum(r["width"] > r["height"] for r in readable)},
        "modes": dict(sorted(modes.items())),
        "mode_groups": {**{m: modes[m] for m in ("RGB", "RGBA", "L")},
                        "other": sum(n for m, n in modes.items() if m not in ("RGB", "RGBA", "L"))},
        "top20_sizes": [{"width": w, "height": h, "count": n}
                        for (w, h), n in sorted(sizes.items(), key=lambda item: (-item[1], item[0]))[:20]],
        "size_thresholds": {"exactly_256x256": sizes[(256, 256)],
                            "short_side_lt256": sum(min(r["width"], r["height"]) < 256 for r in readable),
                            "long_side_lt256": sum(max(r["width"], r["height"]) < 256 for r in readable),
                            "short_side_ge256": sum(min(r["width"], r["height"]) >= 256 for r in readable)},
        "aspect_ratio_buckets": [{"range": label, "count": sum(test(r) for r in ratios),
                                  "percent": sum(test(r) for r in ratios) / len(ratios) * 100 if ratios else 0}
                                 for label, test in buckets],
    }


def cell(value):
    if isinstance(value, (list, dict)):
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            return value[0]
        if value == []:
            return ""
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def table(rows: list[dict], fields: list[str]) -> str:
    def escape(value):
        return str(cell(value)).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")
    return "\n".join([
        "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |",
        *("| " + " | ".join(escape(row.get(f)) for f in fields) + " |" for row in rows),
    ])


def render_report(summary: dict) -> str:
    parts = ["# PAD-UFES-20 原始数据审计", f"状态：`{summary['status']}`。",
             "## 输入与统计口径", table([summary["inputs"]], ["metadata", "image_dir"]),
             *[f"- {rule}" for rule in COUNTING_RULES],
             "## metadata 表头", json.dumps(summary["schema"], ensure_ascii=False, indent=2)]
    if "metadata" in summary:
        meta, images = summary["metadata"], summary["images"]
        parts += ["## metadata 完整性", table(
            [{"metric": k, "value": v} for k, v in meta.items() if k != "hierarchy"], ["metric", "value"]),
            "## 图片文件与属性", table(
                [{"metric": k, "value": images[k]} for k in (
                    "inventoried_image_files", "readable_image_files", "unreadable_image_files",
                    "orientation", "modes", "mode_groups", "size_thresholds")], ["metric", "value"]),
            table([{"attribute": f, **images[f]} for f in ("width", "height", "aspect_ratio", "file_size")],
                  ["attribute", "count", "min", "max", "mean", "median", "p05", "p25", "p75", "p95"]),
            "## 最常见尺寸（前 20）", table(images["top20_sizes"], ["width", "height", "count"]),
            "## 长宽比分桶", table(images["aspect_ratio_buckets"], ["range", "count", "percent"]),
            "## 患者与病灶层级", table(
                [{"metric": k, **meta["hierarchy"][k]} for k in (
                    "images_per_patient", "lesions_per_patient", "images_per_lesion")],
                ["metric", "count", "min", "max", "mean", "median", "p90", "p95", "p99"]),
            "### 图片数最多的 20 位患者", table(meta["hierarchy"]["top20_patients_by_images"],
                                                    ["patient_id", "image_count", "lesion_count", "diagnostics"]),
            "### 病灶数最多的 20 位患者", table(meta["hierarchy"]["top20_patients_by_lesions"],
                                                    ["patient_id", "image_count", "lesion_count", "diagnostics"]),
            "## diagnostic 分布", "空 diagnostic 行代表缺失；未知类别保留原文。",
            table(summary["diagnostics"], ["diagnostic", "image_count", "image_percent", "lesion_count", "patient_count"])]
        parts += ["## lesion_id 标识语义",
                  "原始 lesion_id 原样保留。跨 patient_id 复用属于标识语义现象，不作为数据错误；本轮未修改原始 metadata。",
                  "病灶按 (patient_id, lesion_id) 计数；仅同一对 ID 内多个非空 diagnostic 列为 lesion diagnostic conflict。",
                  "复用明细完整保存在 JSON 的 observations.reused_lesion_ids_across_patients 中。",
                  "## RGBA alpha 审计", table(
                      [{"metric": k, "value": v} for k, v in summary["rgba_alpha"].items()
                       if k != "top20_by_non_opaque_ratio"], ["metric", "value"]),
                  "### alpha<255 像素占比最高的图片（最多 20，仅列存在透明像素者）",
                  table(summary["rgba_alpha"]["top20_by_non_opaque_ratio"],
                        ["img_id", "non_opaque_ratio", "alpha_min", "alpha_max",
                         "non_opaque_pixel_count", "fully_transparent_pixel_count",
                         "partially_transparent_pixel_count", "total_pixel_count"]),
                  "## 几何异常完整列表"]
        for name, entries in summary["geometry"].items():
            parts += [f"### {name}：{len(entries)} 张", table(
                entries, ["img_id", "path", "width", "height", "aspect_ratio", "short_side"])]
    parts += ["## 异常与待核查项", "多诊断患者仅为组成事实，不自动判为数据错误。全部明细见 audit_summary.json；下列每项最多显示 20 条。"]
    for name, entries in summary.get("findings", {}).items():
        count = sum(len(v) for v in entries.values()) if name == "missing_values" else len(entries)
        preview = dict(list(entries.items())[:20]) if isinstance(entries, dict) else entries[:20]
        if name == "missing_values":
            preview = {k: v[:20] for k, v in entries.items()}
        parts += [f"### {name}：{count}", "```json\n" + json.dumps(preview, ensure_ascii=False, indent=2) + "\n```"]
    parts += ["## 本轮边界", "只读输入；没有修改原图，没有生成正式 train/val/test，没有决定 preprocessing，没有启动训练。合成测试不是研究结果。"]
    return "\n\n".join(parts) + "\n"


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({f: cell(row.get(f)) for f in fields} for row in rows)


def audit(metadata: Path, image_dir: Path, output_dir: Path) -> dict:
    metadata, image_dir, output_dir = (Path(p).resolve() for p in (metadata, image_dir, output_dir))
    if not metadata.is_file():
        raise ValueError(f"metadata 文件不存在：{metadata}")
    if not image_dir.is_dir():
        raise ValueError(f"图片目录不存在：{image_dir}")
    if output_dir == image_dir or output_dir.is_relative_to(image_dir):
        raise ValueError("输出目录必须位于原始图片目录之外")
    if output_dir == metadata or metadata.is_relative_to(output_dir):
        raise ValueError("输出目录不能包含原始 metadata")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("输出目录必须不存在或为空；拒绝覆盖已有文件")
    _, rows, schema = read_metadata(metadata)
    summary = {"audit_version": 2, "inputs": {"metadata": str(metadata), "image_dir": str(image_dir)},
               "counting_rules": COUNTING_RULES, "schema": schema}
    if any(schema[k] for k in ("missing_required_fields", "duplicate_fields", "malformed_records")):
        summary["status"] = "invalid_metadata_schema"
    else:
        meta, patients, lesions, diagnostics, findings, observations = metadata_audit(rows)
        files, scan_findings = inventory(image_dir)
        properties, file_findings = inspect_images(files, rows)
        findings.update(scan_findings)
        findings.update(file_findings)
        if not rows:
            findings["empty_metadata"] = ["metadata 没有数据记录"]
        if not properties:
            findings["no_images"] = ["图片目录没有可识别或被引用的图片文件"]
        alpha = alpha_statistics(properties)
        geometry = geometry_lists(properties)
        has_findings = any(any(v.values()) if isinstance(v, dict) else bool(v) for v in findings.values())
        has_findings |= bool(alpha["rgba_images_with_any_transparency"] or any(geometry.values()))
        status = "complete_with_findings" if has_findings else "complete_clean"
        if scan_findings["skipped_directory_symlinks"] or scan_findings["non_regular_entries"]:
            status = "incomplete"
        summary.update(status=status,
                       metadata=meta, images=image_statistics(properties), diagnostics=diagnostics,
                       scanned_file_count=len(files), findings=findings, observations=observations,
                       rgba_alpha=alpha, geometry=geometry)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting inputs or any existing report.
    with (output_dir / "audit_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (output_dir / "audit_report.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(summary))
    if "metadata" in summary:
        write_csv(output_dir / "image_properties.csv", properties, IMAGE_FIELDS)
        write_csv(output_dir / "patient_summary.csv", patients, ("patient_id", "image_count", "lesion_count", "diagnostics"))
        write_csv(output_dir / "lesion_summary.csv", lesions,
                  ("patient_id", "lesion_id", "lesion_key", "image_count", "diagnostic"))
        write_csv(output_dir / "diagnostic_summary.csv", diagnostics,
                  ("diagnostic", "image_count", "image_percent", "lesion_count", "patient_count"))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = audit(args.metadata, args.image_dir, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"审计未完成：{error}", file=sys.stderr)
        return 2
    print(f"{summary['status']}: {args.output_dir.resolve() / 'audit_report.md'}")
    if summary["status"] == "invalid_metadata_schema":
        print("metadata 结构不符，停止对应逻辑。真实字段：" + json.dumps(summary["schema"], ensure_ascii=False), file=sys.stderr)
        return 2
    # Unknown/invalid/incomplete states must never fall through to success.
    return EXIT_CODES.get(summary["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
