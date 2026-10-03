import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from tools.audit_pad_ufes20 import REQUIRED, audit, describe


def dataset(tmp_path, rows, fields=REQUIRED):
    source = tmp_path / "raw"
    source.mkdir()
    images = source / "images"
    images.mkdir()
    metadata = source / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return metadata, images, tmp_path / "audit"


def picture(images, name, size=(256, 256), mode="RGB", fmt="PNG"):
    path = images / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path, format=fmt)
    return path


def read_csv(output, filename):
    with (output / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_normal_cli_and_read_only_inputs(tmp_path):
    metadata, images, output = dataset(tmp_path, [
        ("001", "l1", "a.png", "ACK"), ("002", "l2", "b.png", "BCC")])
    picture(images, "nested/a.png")
    picture(images, "b.png", (300, 300))
    before = fingerprints(metadata.parent)
    script = Path(__file__).resolve().parents[1] / "tools/audit_pad_ufes20.py"
    run = subprocess.run([sys.executable, str(script), "--metadata", str(metadata),
                          "--image-dir", str(images), "--output-dir", str(output)],
                         cwd=tmp_path, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert fingerprints(metadata.parent) == before
    assert {p.name for p in output.iterdir()} == {
        "audit_summary.json", "image_properties.csv", "patient_summary.csv",
        "lesion_summary.csv", "diagnostic_summary.csv", "audit_report.md"}
    summary = json.loads((output / "audit_summary.json").read_text())
    assert summary["status"] == "complete_clean"
    assert summary["metadata"]["total_rows"] == 2
    assert summary["metadata"]["unique_patient_ids"] == 2
    assert summary["images"]["width"]["mean"] == 278
    assert read_csv(output, "patient_summary.csv")[0]["patient_id"] == "001"
    ack = next(r for r in summary["diagnostics"] if r["diagnostic"] == "ACK")
    assert ack == {"diagnostic": "ACK", "image_count": 1, "image_percent": 50,
                   "lesion_count": 1, "patient_count": 1}
    assert "没有决定 preprocessing" in (output / "audit_report.md").read_text()


def test_missing_extra_duplicate_ids_and_filename_ambiguity(tmp_path):
    args = dataset(tmp_path, [("p1", "l1", "a.png", "ACK"),
                              ("p1", "l1", "a.png", "ACK"),
                              ("p2", "l2", "missing.png", "NEV")])
    picture(args[1], "one/a.png")
    picture(args[1], "two/a.png")
    picture(args[1], "extra.png")
    result = audit(*args)
    findings = result["findings"]
    assert result["status"] == "complete_with_findings"
    assert result["metadata"]["total_rows"] == 3
    assert result["metadata"]["unique_img_ids"] == 2
    assert findings["missing_images"] == ["missing.png"]
    assert findings["duplicate_img_ids"] == [{"img_id": "a.png", "record_numbers": [1, 2]}]
    assert len(findings["duplicate_filenames"]["a.png"]) == 2
    assert len(findings["ambiguous_image_matches"]) == 1
    assert findings["unreferenced_pngs"] == [str(args[1] / "extra.png")]
    assert result["images"]["readable_image_files"] == 3
    assert read_csv(args[2], "patient_summary.csv")[0]["image_count"] == "1"


def test_hierarchy_conflicts_and_multiple_diagnostics(tmp_path):
    rows = [("p1", "l1", "a.png", "ACK"), ("p1", "l1", "b.png", "BCC"),
            ("p1", "l2", "c.png", "ACK"), ("p2", "l1", "d.png", "MEL"),
            ("p2", "l1", "a.png", "SCC")]
    args = dataset(tmp_path, rows)
    for name in ("a.png", "b.png", "c.png", "d.png"):
        picture(args[1], name)
    result = audit(*args)
    levels, findings = result["metadata"]["hierarchy"], result["findings"]
    assert levels["images_per_patient"]["min"] == 2
    assert levels["images_per_patient"]["max"] == 3
    assert levels["images_per_patient"]["mean"] == 2.5
    assert levels["images_per_patient"]["p90"] == pytest.approx(2.9)
    assert levels["lesions_per_patient"]["max"] == 2
    assert levels["images_per_lesion"]["max"] == 2
    assert levels["top20_patients_by_images"][0]["patient_id"] == "p1"
    assert levels["top20_patients_by_lesions"][0]["patient_id"] == "p1"
    assert len(findings["patients_with_multiple_diagnostics"]) == 2
    assert result["observations"]["reused_lesion_ids_across_patients"][0]["patient_ids"] == ["p1", "p2"]
    assert len(findings["lesion_diagnostic_conflicts"]) == 2
    assert result["metadata"]["unique_patient_lesion_pairs"] == 3
    assert result["metadata"]["lesion_diagnostic_conflict_count"] == 2
    assert findings["images_with_conflicting_metadata"][0]["img_id"] == "a.png"
    image = next(r for r in read_csv(args[2], "image_properties.csv") if r["img_id"] == "a.png")
    assert json.loads(image["patient_id"]) == ["p1", "p2"]
    assert json.loads(image["diagnostic"]) == ["ACK", "SCC"]
    assert json.loads(image["lesion_keys"]) == [["p1", "l1"], ["p2", "l1"]]
    lesion = read_csv(args[2], "lesion_summary.csv")[0]
    assert lesion["patient_id"] == "p1"
    assert json.loads(lesion["lesion_key"]) == ["p1", "l1"]


def test_dimensions_modes_quantiles_and_size_thresholds(tmp_path):
    args = dataset(tmp_path, [("p", "l", f"{i}.png", "ACK") for i in range(4)])
    for i, (size, mode) in enumerate([((256, 256), "RGB"), ((100, 200), "RGBA"),
                                     ((500, 250), "L"), ((300, 400), "P")]):
        picture(args[1], f"{i}.png", size, mode)
    result = audit(*args)["images"]
    assert result["width"] == {"count": 4, "min": 100, "max": 500, "mean": 289,
                                "median": 278, "p05": pytest.approx(123.4),
                                "p25": 217, "p75": 350, "p95": pytest.approx(470)}
    assert result["height"]["median"] == 253
    assert result["aspect_ratio"]["median"] == .875
    assert result["orientation"] == {"portrait": 2, "square": 1, "landscape": 1}
    assert result["mode_groups"] == {"RGB": 1, "RGBA": 1, "L": 1, "other": 1}
    assert result["modes"]["P"] == 1
    assert result["size_thresholds"] == {"exactly_256x256": 1, "short_side_lt256": 2,
                                          "long_side_lt256": 1, "short_side_ge256": 2}
    assert result["file_size"]["min"] > 0


def test_ratio_bucket_boundaries(tmp_path):
    widths = [49, 50, 74, 75, 89, 90, 110, 111, 133, 134, 200, 201]
    args = dataset(tmp_path, [("p", "l", f"{w}.png", "ACK") for w in widths])
    for w in widths:
        picture(args[1], f"{w}.png", (w, 100))
    buckets = audit(*args)["images"]["aspect_ratio_buckets"]
    assert [b["count"] for b in buckets] == [1, 2, 2, 2, 2, 2, 1]
    assert sum(b["percent"] for b in buckets) == pytest.approx(100)


def test_unknown_and_missing_values_preserved(tmp_path):
    args = dataset(tmp_path, [("", "l", "a.png", "mystery"), ("p", "", "b.png", "ack"),
                              ("p", "l", "", ""), ("  ", " l ", "c.png", " ACK ")])
    for name in ("a.png", "b.png", "c.png"):
        picture(args[1], name)
    result = audit(*args)
    assert result["metadata"]["missing_counts"] == {
        "patient_id": 2, "lesion_id": 1, "img_id": 1, "diagnostic": 1}
    assert result["findings"]["unknown_diagnostics"] == [" ACK ", "ack", "mystery"]
    assert len(result["findings"]["surrounding_whitespace"]) == 2
    assert result["metadata"]["raw_unique_lesion_ids"] == 2
    assert result["metadata"]["unique_patient_lesion_pairs"] == 1
    assert next(r for r in result["diagnostics"] if r["diagnostic"] == "ACK")["image_count"] == 0


def test_corrupt_and_truncated_images(tmp_path):
    args = dataset(tmp_path, [("p", "l", "bad.png", "ACK"), ("p", "l", "truncated.png", "ACK")])
    (args[1] / "bad.png").write_bytes(b"not an image")
    truncated = picture(args[1], "truncated.png", (300, 400))
    truncated.write_bytes(truncated.read_bytes()[:60])
    result = audit(*args)
    assert len(result["findings"]["decode_errors"]) == 2
    assert result["images"]["readable_image_files"] == 0
    assert result["images"]["width"]["mean"] is None
    assert all(r["width"] == "" for r in read_csv(args[2], "image_properties.csv"))


def test_case_extension_and_format_anomalies_no_automatic_matching(tmp_path):
    args = dataset(tmp_path, [("p", "l", "case.png", "ACK"), ("p", "l", "fake.png", "ACK")])
    picture(args[1], "case.PNG")
    picture(args[1], "hidden.data")
    picture(args[1], "fake.png", fmt="JPEG")
    (args[1] / "notes.txt").write_text("hello")
    result = audit(*args)
    findings = result["findings"]
    assert findings["missing_images"] == ["case.png"]
    assert findings["case_mismatch_candidates"][0]["img_id"] == "case.png"
    assert len(findings["extension_anomalies"]) == 2
    assert len(findings["unreferenced_pngs"]) == 2
    assert findings["format_anomalies"][0]["format"] == "JPEG"
    assert len(findings["non_image_files"]) == 1
    assert result["images"]["readable_image_files"] == 3


@pytest.mark.parametrize("fields,rows", [
    (("patient", "lesion_id", "img_id", "diagnostic"), [("p", "l", "a.png", "ACK")]),
    ((*REQUIRED, "patient_id"), [("p", "l", "a.png", "ACK", "q")]),
    (REQUIRED, [("p", "l", "a.png")]),
])
def test_schema_mismatch_stops_logic_and_reports_actual_fields(tmp_path, fields, rows):
    args = dataset(tmp_path, rows, fields)
    result = audit(*args)
    assert result["status"] == "invalid_metadata_schema"
    assert result["schema"]["actual_fields"] == list(fields)
    assert "metadata" not in result
    assert {p.name for p in args[2].iterdir()} == {"audit_summary.json", "audit_report.md"}


def test_empty_metadata_and_directory(tmp_path):
    args = dataset(tmp_path, [])
    result = audit(*args)
    assert result["metadata"]["total_rows"] == 0
    assert result["images"]["width"]["p95"] is None
    assert all(r["image_percent"] == 0 for r in result["diagnostics"])
    assert result["status"] == "complete_with_findings"
    assert read_csv(args[2], "image_properties.csv") == []


def test_reject_outputs_in_raw_tree_and_existing_reports(tmp_path):
    metadata, images, output = dataset(tmp_path, [("p", "l", "a.png", "ACK")])
    picture(images, "a.png")
    before = fingerprints(metadata.parent)
    for unsafe in (images, images / "audit", metadata, metadata.parent):
        with pytest.raises(ValueError):
            audit(metadata, images, unsafe)
    alias = tmp_path / "alias"
    alias.symlink_to(images, target_is_directory=True)
    with pytest.raises(ValueError):
        audit(metadata, images, alias / "audit")
    audit(metadata, images, output)
    report_before = fingerprints(output)
    with pytest.raises(ValueError, match="拒绝覆盖"):
        audit(metadata, images, output)
    assert fingerprints(output) == report_before
    assert fingerprints(metadata.parent) == before


def test_top20_limit_and_deterministic_size_ranking(tmp_path):
    args = dataset(tmp_path, [(f"p{i:02}", f"l{i}", f"{i}.png", "NEV") for i in range(22)])
    for i in range(22):
        picture(args[1], f"{i}.png", (10 + i, 10))
    picture(args[1], "extra.png", (31, 10))
    result = audit(*args)
    top = result["metadata"]["hierarchy"]["top20_patients_by_images"]
    assert len(top) == 20 and top[0]["patient_id"] == "p00"
    sizes = result["images"]["top20_sizes"]
    assert len(sizes) == 20
    assert sizes[0] == {"width": 31, "height": 10, "count": 2}
    assert sizes[1] == {"width": 10, "height": 10, "count": 1}


def test_quantiles_single_and_empty():
    assert describe([7], (90, 95, 99)) == {
        "count": 1, "min": 7, "max": 7, "mean": 7, "median": 7,
        "p90": 7, "p95": 7, "p99": 7}
    assert describe([])["min"] is None


def test_decoder_exception_does_not_stop_remaining_images(tmp_path, monkeypatch):
    args = dataset(tmp_path, [("p", "l", "a.png", "ACK"), ("p", "l", "b.png", "ACK")])
    picture(args[1], "a.png")
    picture(args[1], "b.png")
    original_open = Image.open

    def broken_decoder(path, *positional, **keywords):
        if Path(path).name == "a.png":
            raise EOFError("synthetic decoder failure")
        return original_open(path, *positional, **keywords)

    monkeypatch.setattr(Image, "open", broken_decoder)
    result = audit(*args)
    assert result["images"]["readable_image_files"] == 1
    assert result["findings"]["decode_errors"][0]["error"].startswith("EOFError:")


def test_directory_symlink_is_reported_without_recursion(tmp_path):
    args = dataset(tmp_path, [("p", "l", "a.png", "ACK")])
    picture(args[1], "a.png")
    (args[1] / "loop").symlink_to(args[1], target_is_directory=True)
    result = audit(*args)
    assert result["status"] == "incomplete"
    assert result["images"]["readable_image_files"] == 1
    assert result["findings"]["skipped_directory_symlinks"] == [str(args[1] / "loop")]


def test_cli_findings_and_schema_exit_codes(tmp_path, capsys):
    from tools.audit_pad_ufes20 import main

    metadata, images, output = dataset(tmp_path, [("p", "l", "missing.png", "ACK")])
    assert main(["--metadata", str(metadata), "--image-dir", str(images),
                 "--output-dir", str(output)]) == 1
    metadata.write_text("patient,lesion,img,diagnosis\np,l,a.png,ACK\n")
    assert main(["--metadata", str(metadata), "--image-dir", str(images),
                 "--output-dir", str(tmp_path / "invalid-audit")]) == 2
    assert '"actual_fields": ["patient", "lesion", "img", "diagnosis"]' in capsys.readouterr().err


def test_reused_raw_lesion_id_is_semantics_not_error(tmp_path):
    args = dataset(tmp_path, [("p1", "001", "a.png", "ACK"), ("p2", "001", "b.png", "ACK"),
                              ("p3", "001", "c.png", "BCC")])
    for name in ("a.png", "b.png", "c.png"):
        picture(args[1], name)
    before = fingerprints(args[0].parent)
    result = audit(*args)
    assert result["status"] == "complete_clean"
    assert result["metadata"]["raw_unique_lesion_ids"] == 1
    assert result["metadata"]["unique_patient_lesion_pairs"] == 3
    assert result["metadata"]["reused_lesion_ids_across_patients"] == 1
    assert result["findings"]["lesion_diagnostic_conflicts"] == []
    assert result["metadata"]["hierarchy"]["images_per_lesion"]["max"] == 1
    assert next(r for r in result["diagnostics"] if r["diagnostic"] == "ACK")["lesion_count"] == 2
    assert [r["lesion_id"] for r in read_csv(args[2], "lesion_summary.csv")] == ["001"] * 3
    assert fingerprints(args[0].parent) == before


def test_rgba_alpha_counts_ranges_ranking_and_read_only(tmp_path):
    args = dataset(tmp_path, [("p", "l", f"{i}.png", "ACK") for i in range(5)])
    patterns = [[255] * 4, [1, 128, 255, 255], [0] * 4, [0, 64, 255, 255]]
    for i, alphas in enumerate(patterns):
        img = Image.new("RGBA", (2, 2))
        img.putdata([(10, 20, 30, a) for a in alphas])
        img.save(args[1] / f"{i}.png")
    picture(args[1], "4.png")
    before = fingerprints(args[0].parent)
    result = audit(*args)
    alpha = result["rgba_alpha"]
    assert alpha["audited_rgba_images"] == 4
    assert alpha["fully_opaque_rgba_images"] == 1
    assert alpha["rgba_images_with_any_transparency"] == 3
    assert alpha["rgba_images_with_partial_alpha"] == 2
    assert alpha["rgba_images_with_alpha_zero"] == 2
    assert alpha["alpha_min_range"] == {"min": 0, "max": 255}
    assert alpha["alpha_max_range"] == {"min": 0, "max": 255}
    assert [r["img_id"] for r in alpha["top20_by_non_opaque_ratio"]] == [["2.png"], ["1.png"], ["3.png"]]
    rows = {r["img_id"]: r for r in read_csv(args[2], "image_properties.csv")}
    assert {f: rows["3.png"][f] for f in (
        "alpha_min", "alpha_max", "alpha_unique_count", "non_opaque_pixel_count",
        "fully_transparent_pixel_count", "partially_transparent_pixel_count",
        "total_pixel_count", "non_opaque_ratio")} == {
            "alpha_min": "0", "alpha_max": "255", "alpha_unique_count": "3",
            "non_opaque_pixel_count": "2", "fully_transparent_pixel_count": "1",
            "partially_transparent_pixel_count": "1", "total_pixel_count": "4", "non_opaque_ratio": "0.5"}
    assert rows["4.png"]["alpha_min"] == ""
    assert fingerprints(args[0].parent) == before


def test_all_opaque_rgba_and_absent_rgba(tmp_path):
    args = dataset(tmp_path, [("p", "l", "a.png", "ACK")])
    Image.new("RGBA", (256, 256), (1, 2, 3, 255)).save(args[1] / "a.png")
    result = audit(*args)
    assert result["status"] == "complete_clean"
    assert result["rgba_alpha"]["fully_opaque_rgba_images"] == 1
    assert result["rgba_alpha"]["top20_by_non_opaque_ratio"] == []
    assert result["rgba_alpha"]["alpha_min_range"] == {"min": 255, "max": 255}
    from tools.audit_pad_ufes20 import alpha_statistics
    empty = alpha_statistics([])
    assert empty["audited_rgba_images"] == 0
    assert empty["alpha_max_range"] == {"min": None, "max": None}


def test_alpha_top20_limit(tmp_path):
    args = dataset(tmp_path, [("p", "l", f"{i:02}.png", "ACK") for i in range(23)])
    for i in range(23):
        picture(args[1], f"{i:02}.png", (2, 2), "RGBA")
    alpha = audit(*args)["rgba_alpha"]
    assert len(alpha["top20_by_non_opaque_ratio"]) == 20
    assert alpha["top20_by_non_opaque_ratio"][0]["img_id"] == ["00.png"]


def test_geometry_lists_include_all_and_respect_boundaries(tmp_path):
    sizes = [(899, 1000), (900, 1000), (1100, 1000), (1101, 1000),
             (255, 300), (256, 256), (256, 300)]
    args = dataset(tmp_path, [("p", "l", f"{i}.png", "ACK") for i in range(len(sizes))])
    for i, size in enumerate(sizes):
        picture(args[1], f"{i}.png", size)
    result = audit(*args)
    assert [r["img_id"] for r in result["geometry"]["aspect_ratio_outside_0_9_to_1_1"]] == [
        ["0.png"], ["3.png"], ["4.png"], ["6.png"]]
    assert [r["img_id"] for r in result["geometry"]["short_side_lt256"]] == [["4.png"]]
    assert "short_side_lt256：1 张" in (args[2] / "audit_report.md").read_text()


@pytest.mark.parametrize("scenario,expected_status,expected_code", [
    ("clean", "complete_clean", 0),
    ("findings", "complete_with_findings", 1),
    ("schema", "invalid_metadata_schema", 2),
    ("invalid", None, 2),
    ("incomplete", "incomplete", 2),
])
def test_real_process_exit_codes(tmp_path, scenario, expected_status, expected_code):
    metadata, images, output = dataset(tmp_path, [("p", "l", "a.png", "ACK")])
    if scenario != "findings":
        picture(images, "a.png")
    if scenario == "schema":
        metadata.write_text("patient,lesion,img,diagnosis\np,l,a.png,ACK\n")
    elif scenario == "invalid":
        metadata = metadata.parent / "absent.csv"
    elif scenario == "incomplete":
        (images / "loop").symlink_to(images, target_is_directory=True)
    script = Path(__file__).resolve().parents[1] / "tools/audit_pad_ufes20.py"
    run = subprocess.run([sys.executable, str(script), "--metadata", str(metadata),
                          "--image-dir", str(images), "--output-dir", str(output)],
                         cwd=tmp_path, text=True, capture_output=True)
    assert run.returncode == expected_code, (run.stdout, run.stderr)
    if expected_status:
        assert json.loads((output / "audit_summary.json").read_text())["status"] == expected_status
