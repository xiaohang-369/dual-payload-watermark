#!/usr/bin/env python3
"""Read-only analysis of the fixed-bank RGB-loss ablation experiments.

The script reads existing metrics/checkpoints and writes only generated analysis
artifacts below runs/joint_10x20_rgb3_v1/analysis.  It never saves a checkpoint
or mutates a training/config/asset file.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "joint_10x20_rgb3_v1" / "analysis"
EXPERIMENTS = {
    "RGB1": (ROOT / "runs" / "joint_10x20_clean_v1_full", 1.0),
    "RGB2": (ROOT / "runs" / "joint_10x20_rgb2_v1", 2.0),
    "RGB3": (ROOT / "runs" / "joint_10x20_rgb3_v1", 3.0),
}

REQUIRED_KEYS = {
    "rgb_psnr", "rgb_ssim", "rgb_psnr_clipped", "rgb_ssim_clipped",
    "ber", "bit_accuracy", "message_accuracy", "carrier_psnr",
    "carrier_ssim", "delta_c_rms", "delta_w_rms", "loss_rgb",
    "loss_message", "loss_carrier", "loss_range", "loss_total",
}


def load_validation(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if row.get("kind") == "validation":
                rows.append(row)
    return sorted(rows, key=lambda row: (int(row["epoch"]), int(row["global_step"])))


def f(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def epoch_link(row: dict[str, Any]) -> str:
    return f"epoch {int(row['epoch'])} / step {int(row['global_step'])}"


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rows = list(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
    return pearson, spearman


def first_crossing(rows: list[dict[str, Any]], key: str, threshold: float, op: str) -> dict[str, Any] | None:
    for row in rows:
        value = float(row[key])
        if ((op == "lt" and value < threshold) or
                (op == "ge" and value >= threshold) or
                (op == "eq" and math.isclose(value, threshold, abs_tol=1e-12))):
            return row
    return None


def integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = [int(row["epoch"]) for row in rows]
    missing_epochs = sorted(set(range(1, 201)) - set(epochs))
    duplicates = sorted({epoch for epoch in epochs if epochs.count(epoch) > 1})
    missing_keys = {
        int(row["epoch"]): sorted(REQUIRED_KEYS - set(row))
        for row in rows if REQUIRED_KEYS - set(row)
    }
    step_mismatches = [
        {"epoch": int(row["epoch"]), "global_step": int(row["global_step"]), "expected": int(row["epoch"]) * 25}
        for row in rows if int(row["global_step"]) != int(row["epoch"]) * 25
    ]
    passed = (
        len(rows) == 200 and epochs and min(epochs) == 1 and max(epochs) == 200
        and not missing_epochs and not duplicates and not missing_keys and not step_mismatches
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "count": len(rows),
        "min_epoch": min(epochs) if epochs else None,
        "max_epoch": max(epochs) if epochs else None,
        "missing_epochs": missing_epochs,
        "duplicate_epochs": duplicates,
        "missing_keys": missing_keys,
        "step_mismatches": step_mismatches,
    }


def milestone_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = [[], [], []]
    specs = [
        ("BER", "ber", [("<", x, "lt") for x in (.40, .30, .20, .10, .05, .01, .005, .001)]),
        ("Exact", "message_accuracy", [(">=", x, "ge") for x in (.50, .80, .90, .95, .99)] + [("=", 1.0, "eq")]),
        ("PSNR", "rgb_psnr", [(">=", x, "ge") for x in (25, 27, 28, 29, 30)]),
    ]
    for output, (_, key, thresholds) in zip(groups, specs):
        for symbol, threshold, op in thresholds:
            row = first_crossing(rows, key, threshold, op)
            output.append({
                "threshold": f"{key} {symbol} {threshold:g}",
                "epoch": None if row is None else int(row["epoch"]),
                "global_step": None if row is None else int(row["global_step"]),
                "rgb_psnr": None if row is None else row["rgb_psnr"],
                "ber": None if row is None else row["ber"],
                "message_accuracy": None if row is None else row["message_accuracy"],
            })
    return groups[0], groups[1], groups[2]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def robust_threshold(values: np.ndarray, floor: float) -> float:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return max(floor, abs(center) + 3.0 * 1.4826 * mad)


def dynamics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = np.array([row["epoch"] for row in rows], dtype=int)
    psnr = np.array([row["rgb_psnr"] for row in rows], dtype=float)
    ber = np.array([row["ber"] for row in rows], dtype=float)
    exact = np.array([row["message_accuracy"] for row in rows], dtype=float)
    d_psnr = np.diff(psnr)
    d_ber = np.diff(ber)

    ber_threshold = robust_threshold(d_ber, 0.01)
    psnr_threshold = robust_threshold(d_psnr, 0.50)
    events: list[dict[str, Any]] = []
    for index in range(1, len(rows)):
        signals: list[str] = []
        if d_ber[index - 1] >= ber_threshold:
            signals.append(f"BER +{d_ber[index - 1]:.6f}")
        if d_psnr[index - 1] <= -psnr_threshold:
            signals.append(f"PSNR {d_psnr[index - 1]:.3f} dB")
        if not signals:
            continue
        recovery = None
        for later in range(index + 1, len(rows)):
            ber_recovered = d_ber[index - 1] < ber_threshold or ber[later] <= ber[index - 1]
            psnr_recovered = d_psnr[index - 1] > -psnr_threshold or psnr[later] >= psnr[index - 1]
            if ber_recovered and psnr_recovered:
                recovery = int(epochs[later])
                break
        events.append({
            "from_epoch": int(epochs[index - 1]), "epoch": int(epochs[index]),
            "signals": ", ".join(signals), "recovery_epoch": recovery,
            "ber": float(ber[index]), "rgb_psnr": float(psnr[index]),
        })
    events.sort(key=lambda event: abs(float(event["ber"] - ber[event["from_epoch"] - 1])) + abs(float(event["rgb_psnr"] - psnr[event["from_epoch"] - 1])) / 10, reverse=True)

    # A plateau is a 10-epoch window whose full range is narrow in both axes.
    plateau_windows: list[tuple[int, int]] = []
    window = 10
    for start in range(0, len(rows) - window + 1):
        stop = start + window
        if np.ptp(psnr[start:stop]) <= 0.50 and np.ptp(ber[start:stop]) <= 0.01:
            plateau_windows.append((int(epochs[start]), int(epochs[stop - 1])))
    plateaus = sorted(merge_intervals(plateau_windows), key=lambda x: (x[1] - x[0], -x[0]), reverse=True)

    rapid_ber_start = first_crossing(rows, "ber", .40, "lt")
    rapid_ber_end = first_crossing(rows, "ber", .05, "lt")
    rapid_psnr_start = first_crossing(rows, "rgb_psnr", 25, "ge")
    rapid_psnr_end = first_crossing(rows, "rgb_psnr", 29, "ge")
    late = rows[-40:]
    late_stats = {
        key: {
            "min": float(min(row[key] for row in late)),
            "max": float(max(row[key] for row in late)),
            "mean": float(np.mean([row[key] for row in late])),
            "std": float(np.std([row[key] for row in late])),
        }
        for key in ("rgb_psnr", "ber", "message_accuracy")
    }
    return {
        "definition": {
            "regression": f"one-epoch BER increase >= {ber_threshold:.6f} or PSNR drop >= {psnr_threshold:.3f} dB (robust 3-MAD rule with floors)",
            "recovery": "first later epoch returning to or beyond the pre-event value for every triggered metric",
            "plateau": "at least one rolling 10-epoch window with PSNR range <= 0.50 dB and BER range <= 0.01",
            "late_window": "epochs 161-200 (descriptive range/std only)",
        },
        "rapid_ber_interval": None if not rapid_ber_start or not rapid_ber_end else [rapid_ber_start["epoch"], rapid_ber_end["epoch"]],
        "rapid_psnr_interval": None if not rapid_psnr_start or not rapid_psnr_end else [rapid_psnr_start["epoch"], rapid_psnr_end["epoch"]],
        "plateaus": plateaus[:5],
        "regressions": events[:8],
        "late_stats": late_stats,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def ticks(low: float, high: float, count: int = 6) -> list[float]:
    if math.isclose(low, high):
        return [low]
    return list(np.linspace(low, high, count))


def draw_vertical_label(image: Image.Image, text: str, y_center: float) -> None:
    label_font = font(27)
    box = label_font.getbbox(text)
    width, height = box[2] - box[0] + 24, box[3] - box[1] + 24
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((width / 2, height / 2), text, font=label_font, fill="#172033", anchor="mm")
    rotated = layer.rotate(90, expand=True)
    image.paste(rotated, (24, int(y_center - rotated.height / 2)), rotated)


def draw_line_plot(
    path: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str,
    markers: list[tuple[int, str, str]], *, log_y: bool = False,
) -> None:
    width, height = 1800, 1100
    left, right, top, bottom = 180, 80, 110, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font, legend_font = font(38, True), font(27), font(21), font(21)
    draw.text((width / 2, 38), title, font=title_font, fill="#172033", anchor="ma")

    plot_x0, plot_x1 = left, width - right
    plot_y0, plot_y1 = top, height - bottom
    x_min, x_max = min(xs), max(xs)
    positive = [y for y in ys if y > 0]
    floor_y = min(positive) / 10 if positive else 1e-8
    transformed = [math.log10(max(y, floor_y)) if log_y else y for y in ys]
    y_min, y_max = min(transformed), max(transformed)
    y_pad = max((y_max - y_min) * .08, 1e-6)
    y_min, y_max = y_min - y_pad, y_max + y_pad
    if not log_y and min(ys) >= 0:
        y_min = max(0.0, y_min)

    def px(x: float) -> float:
        return plot_x0 + (x - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)

    def py(y: float) -> float:
        value = math.log10(max(y, floor_y)) if log_y else y
        return plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)

    for value in ticks(x_min, x_max, 9):
        x = px(value)
        draw.line((x, plot_y0, x, plot_y1), fill="#e4e8ef", width=1)
        draw.text((x, plot_y1 + 18), f"{value:.0f}", font=tick_font, fill="#4c566a", anchor="ma")
    for value in ticks(y_min, y_max, 7):
        y = plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)
        draw.line((plot_x0, y, plot_x1, y), fill="#e4e8ef", width=1)
        label = f"10^{value:.1f}" if log_y else (f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}")
        draw.text((plot_x0 - 18, y), label, font=tick_font, fill="#4c566a", anchor="rm")
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#2f3542", width=3)
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#2f3542", width=3)
    points = [(px(x), py(y)) for x, y in zip(xs, ys)]
    draw.line(points, fill="#2563eb", width=5, joint="curve")
    for x, y in points[::5]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#2563eb")

    seen: set[str] = set()
    legend: list[tuple[str, str]] = []
    for index, label, color in markers:
        x, y = points[index]
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="white", outline=color, width=5)
        if label not in seen:
            legend.append((label, color))
            seen.add(label)
    if legend:
        box_w, box_h = 390, 44 * len(legend) + 26
        x0, y0 = plot_x1 - box_w - 12, plot_y0 + 12
        draw.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=12, fill="#ffffffee", outline="#c8cfda", width=2)
        for idx, (label, color) in enumerate(legend):
            y = y0 + 24 + idx * 44
            draw.ellipse((x0 + 18, y - 8, x0 + 34, y + 8), fill="white", outline=color, width=4)
            draw.text((x0 + 48, y), label, font=legend_font, fill="#172033", anchor="lm")
    draw.text(((plot_x0 + plot_x1) / 2, height - 58), xlabel, font=label_font, fill="#172033", anchor="ma")
    draw_vertical_label(image, ylabel, (plot_y0 + plot_y1) / 2)
    image.save(path, format="PNG", dpi=(180, 180))


def draw_trajectory(path: Path, rows: list[dict[str, Any]], markers: list[tuple[int, str, str]]) -> None:
    xs = [float(row["ber"]) for row in rows]
    ys = [float(row["rgb_psnr"]) for row in rows]
    width, height = 1800, 1100
    left, right, top, bottom = 180, 80, 110, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2, 38), "RGB3 PSNR-BER Training Trajectory", font=font(38, True), fill="#172033", anchor="ma")
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xp, yp = (xmax - xmin) * .06, (ymax - ymin) * .06
    xmin, xmax, ymin, ymax = xmin - xp, xmax + xp, ymin - yp, ymax + yp
    if min(xs) >= 0:
        xmin = max(0.0, xmin)

    def px(v: float) -> float: return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    def py(v: float) -> float: return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    for value in ticks(xmin, xmax, 7):
        x = px(value); draw.line((x, y0, x, y1), fill="#e4e8ef")
        draw.text((x, y1 + 18), f"{value:.3f}", font=font(21), fill="#4c566a", anchor="ma")
    for value in ticks(ymin, ymax, 7):
        y = py(value); draw.line((x0, y, x1, y), fill="#e4e8ef")
        draw.text((x0 - 18, y), f"{value:.2f}", font=font(21), fill="#4c566a", anchor="rm")
    draw.line((x0, y0, x0, y1), fill="#2f3542", width=3); draw.line((x0, y1, x1, y1), fill="#2f3542", width=3)
    points = [(px(x), py(y)) for x, y in zip(xs, ys)]
    for i in range(len(points) - 1):
        ratio = i / max(1, len(points) - 2)
        color = (int(37 + 210 * ratio), int(99 - 45 * ratio), int(235 - 170 * ratio))
        draw.line((points[i], points[i + 1]), fill=color, width=4)
    for i, (x, y) in enumerate(points):
        if i % 5 == 0:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#374151")
    label_offsets = {
        "Highest PSNR": (18, -20), "Final": (18, 3),
        "Lowest BER": (18, -8), "First BER=0": (18, 18),
    }
    for index, label, color in markers:
        x, y = points[index]
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="white", outline=color, width=5)
        dx, dy = label_offsets.get(label, (18, 0))
        draw.text((x + dx, y + dy), label, font=font(18, True), fill=color, anchor="lm")
    draw.text(((x0 + x1) / 2, height - 58), "BER", font=font(27), fill="#172033", anchor="ma")
    draw_vertical_label(image, "RGB PSNR (dB)", (y0 + y1) / 2)
    draw.text((x1 - 8, y1 + 72), "blue = early, red = late", font=font(20), fill="#4c566a", anchor="ra")
    image.save(path, format="PNG", dpi=(180, 180))


def checkpoint_metadata(run_dir: Path) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        return [{"file": name, "error": f"torch unavailable: {exc}"} for name in ("best_message_ber.pt", "best_exact_success.pt", "last.pt")]
    result = []
    for name in ("best_message_ber.pt", "best_exact_success.pt", "last.pt"):
        path = run_dir / name
        if not path.exists():
            result.append({"file": name, "exists": False})
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            step = int(payload.get("global_step", -1))
            validation = payload.get("validation") or {}
            result.append({
                "file": name, "exists": True, "global_step": step,
                "epoch": step // 25 if step >= 0 and step % 25 == 0 else None,
                "next_epoch": payload.get("next_epoch"),
                "gradient_accumulation_steps": payload.get("gradient_accumulation_steps"),
                "ber": validation.get("ber"), "message_accuracy": validation.get("message_accuracy"),
                "rgb_psnr": validation.get("rgb_psnr"),
            })
        except Exception as exc:  # Report a read error without modifying the file.
            result.append({"file": name, "exists": True, "error": f"{type(exc).__name__}: {exc}"})
    return result


def selected_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = ["epoch", "global_step", "rgb_psnr", "rgb_ssim", "ber", "bit_accuracy", "message_accuracy", "loss_message", "loss_rgb"]
    return {key: row.get(key) for key in fields}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {name: load_validation(path) for name, (path, _) in EXPERIMENTS.items()}
    rgb3 = data["RGB3"]
    audit = integrity(rgb3)
    if not rgb3:
        raise RuntimeError("RGB3 has no validation records")

    final = rgb3[-1]
    best_psnr = max(rgb3, key=lambda row: (row["rgb_psnr"], -row["epoch"]))
    best_ssim = max(rgb3, key=lambda row: (row["rgb_ssim"], -row["epoch"]))
    best_ber = min(rgb3, key=lambda row: (row["ber"], -row["rgb_psnr"], row["epoch"]))
    max_exact = max(row["message_accuracy"] for row in rgb3)
    exact_ties = [row for row in rgb3 if math.isclose(row["message_accuracy"], max_exact, abs_tol=1e-12)]
    zero_ber = [row for row in rgb3 if math.isclose(row["ber"], 0.0, abs_tol=1e-12)]
    exact_one = [row for row in rgb3 if math.isclose(row["message_accuracy"], 1.0, abs_tol=1e-12)]
    watermark_optimal = max(zero_ber, key=lambda row: (row["rgb_psnr"], -row["epoch"])) if zero_ber else best_ber

    ber_milestones, exact_milestones, psnr_milestones = milestone_rows(rgb3)
    dyn = dynamics(rgb3)
    pearson, spearman = correlation(
        np.array([row["rgb_psnr"] for row in rgb3], dtype=float),
        np.array([row["ber"] for row in rgb3], dtype=float),
    )

    epoch_maps = {name: {int(row["epoch"]): row for row in rows} for name, rows in data.items()}
    common_epochs = sorted(set.intersection(*(set(mapping) for mapping in epoch_maps.values())))
    common_rows: list[dict[str, Any]] = []
    for epoch in common_epochs:
        for name in ("RGB1", "RGB2", "RGB3"):
            row = epoch_maps[name][epoch]
            common_rows.append({
                "epoch": epoch, "experiment": name, "rgb_loss": EXPERIMENTS[name][1],
                "rgb_psnr": row["rgb_psnr"], "rgb_ssim": row["rgb_ssim"], "ber": row["ber"],
                "bit_accuracy": row["bit_accuracy"], "message_accuracy": row["message_accuracy"],
                "carrier_psnr": row["carrier_psnr"],
            })
    final_rows = []
    for name in ("RGB1", "RGB2", "RGB3"):
        row = epoch_maps[name].get(200)
        if row is None:
            continue
        final_rows.append({
            "experiment": name, "rgb_loss": EXPERIMENTS[name][1], "epoch": 200,
            "rgb_psnr": row["rgb_psnr"], "rgb_ssim": row["rgb_ssim"], "ber": row["ber"],
            "bit_accuracy": row["bit_accuracy"], "message_accuracy": row["message_accuracy"],
            "carrier_psnr": row["carrier_psnr"],
        })
    final_by_name = {row["experiment"]: row for row in final_rows}
    deltas = []
    for source, target in (("RGB1", "RGB2"), ("RGB2", "RGB3"), ("RGB1", "RGB3")):
        deltas.append({
            "comparison": f"{source}->{target}",
            **{key: final_by_name[target][key] - final_by_name[source][key]
               for key in ("rgb_psnr", "rgb_ssim", "ber", "bit_accuracy", "message_accuracy", "carrier_psnr")},
        })

    checkpoints = checkpoint_metadata(EXPERIMENTS["RGB3"][0])
    marker_indices: list[tuple[int, str, str]] = []
    def add_marker(row: dict[str, Any], label: str, color: str) -> None:
        marker_indices.append((rgb3.index(row), label, color))
    add_marker(final, "Final", "#111827")
    add_marker(best_ber, "Lowest BER", "#16a34a")
    add_marker(best_psnr, "Highest PSNR", "#dc2626")
    if zero_ber:
        add_marker(zero_ber[0], "First BER=0", "#7c3aed")

    epochs = [row["epoch"] for row in rgb3]
    draw_line_plot(OUT / "rgb_psnr_vs_epoch.png", epochs, [row["rgb_psnr"] for row in rgb3], "RGB3 RGB PSNR vs Epoch", "Epoch", "RGB PSNR (dB)", marker_indices)
    draw_line_plot(OUT / "ber_vs_epoch.png", epochs, [row["ber"] for row in rgb3], "RGB3 BER vs Epoch", "Epoch", "BER", marker_indices)
    draw_line_plot(OUT / "ber_vs_epoch_log.png", epochs, [row["ber"] for row in rgb3], "RGB3 BER vs Epoch (Log Scale)", "Epoch", "BER (log10; zeros at plotting floor)", marker_indices, log_y=True)
    draw_line_plot(OUT / "message_accuracy_vs_epoch.png", epochs, [row["message_accuracy"] for row in rgb3], "RGB3 Exact Message Accuracy vs Epoch", "Epoch", "Message accuracy", marker_indices)
    draw_trajectory(OUT / "psnr_vs_ber.png", rgb3, marker_indices)

    validation_fields = ["epoch", "global_step"] + sorted(REQUIRED_KEYS)
    write_csv(OUT / "rgb3_validation_metrics.csv", rgb3, validation_fields)
    write_csv(OUT / "common_epoch_comparison.csv", common_rows, list(common_rows[0]))
    write_csv(OUT / "final_epoch_comparison.csv", final_rows, list(final_rows[0]))
    write_csv(OUT / "milestones_ber.csv", ber_milestones, list(ber_milestones[0]))
    write_csv(OUT / "milestones_exact.csv", exact_milestones, list(exact_milestones[0]))
    write_csv(OUT / "milestones_psnr.csv", psnr_milestones, list(psnr_milestones[0]))

    analysis_json = {
        "protocol": {"images": 10, "messages": 20, "pairs": 200, "train_validation_relation": "same fixed 200-pair bank", "epochs": 200, "updates_per_epoch": 25},
        "integrity": audit,
        "rgb3": {
            "final": selected_row(final), "best_psnr": selected_row(best_psnr), "best_ssim": selected_row(best_ssim),
            "best_ber": selected_row(best_ber), "max_message_accuracy": max_exact,
            "max_message_accuracy_epochs": [row["epoch"] for row in exact_ties],
            "zero_ber_epochs": [row["epoch"] for row in zero_ber],
            "exact_one_epochs": [row["epoch"] for row in exact_one],
            "watermark_optimal": selected_row(watermark_optimal),
        },
        "milestones": {"ber": ber_milestones, "exact": exact_milestones, "psnr": psnr_milestones},
        "dynamics": dyn,
        "correlation": {"pearson_psnr_ber": pearson, "spearman_psnr_ber": spearman},
        "common_epochs": common_epochs,
        "final_epoch": final_rows,
        "final_deltas": deltas,
        "checkpoints": checkpoints,
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis_json, ensure_ascii=False, indent=2), encoding="utf-8")

    def milestone_md(items: list[dict[str, Any]]) -> str:
        return markdown_table(
            ["Threshold", "Epoch", "Step", "PSNR", "BER", "Exact"],
            [[item["threshold"], item["epoch"] if item["epoch"] is not None else "not reached", item["global_step"] or "-", f(item["rgb_psnr"]), f(item["ber"]), f(item["message_accuracy"])] for item in items],
        )

    op_rows = [("Watermark-optimal", watermark_optimal), ("RGB-optimal", best_psnr), ("Final", final)]
    focus_epochs = [epoch for epoch in (50, 100, 150, 200) if epoch in common_epochs]
    common_focus = []
    for epoch in focus_epochs:
        for name in ("RGB1", "RGB2", "RGB3"):
            row = epoch_maps[name][epoch]
            common_focus.append([epoch, name, f(row["rgb_psnr"]), f(row["rgb_ssim"]), f(row["ber"]), f(row["bit_accuracy"]), f(row["message_accuracy"]), f(row["carrier_psnr"])])

    rgb_trend = [final_by_name[name]["rgb_psnr"] for name in ("RGB1", "RGB2", "RGB3")]
    ber_trend = [final_by_name[name]["ber"] for name in ("RGB1", "RGB2", "RGB3")]
    exact_trend = [final_by_name[name]["message_accuracy"] for name in ("RGB1", "RGB2", "RGB3")]
    psnr_monotonic = rgb_trend[0] < rgb_trend[1] < rgb_trend[2]
    ber_monotonic_worse = ber_trend[0] < ber_trend[1] < ber_trend[2]

    report: list[str] = []
    report += ["# RGB Loss Weight Ablation Analysis", "",
        "## 1. Experiment Protocol", "",
        "This analysis uses the fixed `joint_10x20_v1` protocol: 10 fixed images crossed with 20 fixed 64-bit messages, producing 200 fixed pairs. The validation set is the same fixed 200-pair bank as training, so all results describe fixed-protocol learnability/memorization behavior rather than generalization to unseen images or messages. The experiment does not establish recovery for arbitrary 64-bit payloads.", "",
        "RGB1/RGB2 have sparse validation records, while RGB3 has per-epoch validation. Cross-experiment comparisons below therefore use only the exact shared validation epochs; epoch 50/100/150/200 are emphasized. RGB3's arbitrary-epoch extrema are analyzed only within RGB3 and are not presented as fair cross-experiment best-vs-best comparisons.", "",
        "The observed `delta_c_rms` / `delta_w_rms` values near `2/255` indicate the RMS perturbation budget is near its cap. They do not mean every pixel is independently bounded to +/-2/255.", "",
        "## 2. RGB3 Metrics Integrity", "",
        f"**{audit['status']}** — validation rows: {audit['count']}; epoch range: {audit['min_epoch']}..{audit['max_epoch']}; missing epochs: {audit['missing_epochs'] or 'none'}; duplicate epochs: {audit['duplicate_epochs'] or 'none'}; rows with missing required keys: {len(audit['missing_keys'])}; global-step mismatches against `epoch * 25`: {len(audit['step_mismatches'])}.", "",
        "Checkpoint metadata was loaded on CPU with `torch.load(..., weights_only=True)`; no checkpoint was saved or changed.", "",
        markdown_table(["File", "Exists", "Epoch from step", "Step", "BER", "Exact", "PSNR", "Accumulation"], [[c.get("file"), c.get("exists", "N/A"), c.get("epoch", "N/A"), c.get("global_step", "N/A"), f(c.get("ber")), f(c.get("message_accuracy")), f(c.get("rgb_psnr")), c.get("gradient_accumulation_steps", "N/A")] for c in checkpoints]), "",
        "## 3. RGB3 Training Dynamics", "",
        "### Data facts", "",
        f"The requested threshold-derived rapid BER interval is epoch {dyn['rapid_ber_interval']} (`BER < 0.40` to first `BER < 0.05`), while the threshold-derived PSNR climb interval is epoch {dyn['rapid_psnr_interval']} (first 25 dB to first 29 dB). These boundaries are computed from the data, not entered as epoch labels.", "",
        f"Plateau rule: {dyn['definition']['plateau']}. Detected longest intervals: {dyn['plateaus'] or 'none'}.", "",
        f"Regression-event rule: {dyn['definition']['regression']}. Recovery rule: {dyn['definition']['recovery']}.", "",
    ]
    if dyn["regressions"]:
        report += [markdown_table(["From", "Event epoch", "Observed change", "PSNR", "BER", "Recovery epoch"], [[event["from_epoch"], event["epoch"], event["signals"], f(event["rgb_psnr"]), f(event["ber"]), event["recovery_epoch"] or "not recovered by 200"] for event in dyn["regressions"]]), ""]
    late = dyn["late_stats"]
    report += [
        f"In the defined late window (epochs 161-200), PSNR spans {f(late['rgb_psnr']['min'])} to {f(late['rgb_psnr']['max'])} dB (std {f(late['rgb_psnr']['std'])}); BER spans {f(late['ber']['min'])} to {f(late['ber']['max'])} (std {f(late['ber']['std'])}); exact-message accuracy spans {f(late['message_accuracy']['min'])} to {f(late['message_accuracy']['max'])} (std {f(late['message_accuracy']['std'])}).", "",
        "### Interpretation / hypotheses", "",
        "The detected one-epoch regressions and recoveries establish non-monotonic validation behavior on the fixed bank. They do not, by themselves, establish model degradation or overfitting. Plausible causes include optimization noise and competition among weighted loss terms, but those mechanisms were not isolated in this experiment.", "",
        "## 4. Key Turning Points", "",
        "### BER milestones", "", milestone_md(ber_milestones), "", "### Exact-message milestones", "", milestone_md(exact_milestones), "", "### RGB PSNR milestones", "", milestone_md(psnr_milestones), "",
        f"All BER=0 epochs: `{[row['epoch'] for row in zero_ber]}`.", "",
        f"All exact-message-accuracy=1 epochs: `{[row['epoch'] for row in exact_one]}`. Maximum exact-message accuracy is {f(max_exact)}, tied at epochs `{[row['epoch'] for row in exact_ties]}`.", "",
        "## 5. RGB3 Best Operating Points", "",
        markdown_table(["Point", "Epoch", "Step", "PSNR", "SSIM", "BER", "Bit acc.", "Exact", "Loss message", "Loss RGB"], [[label, row["epoch"], row["global_step"], f(row["rgb_psnr"]), f(row["rgb_ssim"]), f(row["ber"]), f(row["bit_accuracy"]), f(row["message_accuracy"]), f(row["loss_message"]), f(row["loss_rgb"])] for label, row in op_rows]), "",
        f"Maximum RGB SSIM occurs at {epoch_link(best_ssim)}: SSIM {f(best_ssim['rgb_ssim'])}, PSNR {f(best_ssim['rgb_psnr'])}, BER {f(best_ssim['ber'])}, exact {f(best_ssim['message_accuracy'])}.", "",
        "When multiple BER=0 rows exist, the watermark-optimal representative is the zero-BER row with the highest RGB PSNR, per the requested rule.", "",
        "## 6. RGB1 vs RGB2 vs RGB3 Common-Epoch Comparison", "",
        f"Exact common validation epochs are `{common_epochs}`. The full machine-readable table is `common_epoch_comparison.csv`; the priority epochs are shown below.", "",
        markdown_table(["Epoch", "Exp", "PSNR", "SSIM", "BER", "Bit acc.", "Exact", "Carrier PSNR"], common_focus), "",
        "These are like-for-like temporal comparisons. RGB3's dense history supplies more within-run detail, but it does not grant additional cross-run comparison points.", "",
        "## 7. Final Epoch Ablation", "",
        markdown_table(["Exp", "RGB loss", "PSNR", "SSIM", "BER", "Bit acc.", "Exact", "Carrier PSNR"], [[row["experiment"], row["rgb_loss"], f(row["rgb_psnr"]), f(row["rgb_ssim"]), f(row["ber"]), f(row["bit_accuracy"]), f(row["message_accuracy"]), f(row["carrier_psnr"])] for row in final_rows]), "",
        "### Final-epoch deltas", "",
        markdown_table(["Comparison", "Delta PSNR", "Delta SSIM", "Delta BER", "Delta bit acc.", "Delta exact", "Delta carrier PSNR"], [[row["comparison"], f(row["rgb_psnr"]), f(row["rgb_ssim"]), f(row["ber"]), f(row["bit_accuracy"]), f(row["message_accuracy"]), f(row["carrier_psnr"])] for row in deltas]), "",
        "## 8. PSNR-BER Relationship", "",
        f"Across RGB3's 200 per-epoch validation observations, Pearson correlation between RGB PSNR and BER is **{pearson:.6f}** and Spearman rank correlation is **{spearman:.6f}**. These are descriptive associations along one optimization trajectory; they do not imply that changing PSNR causes BER to change.", "",
        "`psnr_vs_ber.png` is a time-ordered training trajectory, not a Pareto frontier. No Pareto-optimality claim is made.", "",
        "## 9. What RGB=3 Tells Us", "",
        "### Data facts", "",
        f"At epoch 200, RGB PSNR values are {[round(x, 6) for x in rgb_trend]} dB for RGB loss 1/2/3. This is {'a strictly increasing' if psnr_monotonic else 'not a strictly increasing'} final-epoch sequence.", "",
        f"At epoch 200, BER values are {[round(x, 8) for x in ber_trend]} and exact-message accuracies are {[round(x, 6) for x in exact_trend]} for RGB loss 1/2/3. BER is {'monotonically worse' if ber_monotonic_worse else 'not monotonically worse'} as RGB loss increases.", "",
        f"RGB3 reaches a within-run minimum BER of {f(best_ber['ber'])} at epoch {best_ber['epoch']} and a within-run maximum PSNR of {f(best_psnr['rgb_psnr'])} dB at epoch {best_psnr['epoch']}; those extrema are not used as cross-run fair-best comparisons.", "",
        "### Interpretation / hypotheses", "",
        "The final-epoch evidence supports stronger RGB reconstruction under RGB loss 3 relative to 1 and 2. The same final-epoch evidence does not support a simple monotonic 'more RGB quality necessarily worsens watermark recovery' tradeoff. Because evaluation reuses the training bank, this conclusion is restricted to the fixed 200-pair protocol.", "",
        "## 10. Current Limitations", "",
        "- Training and validation use the same fixed 200 pairs; there is no held-out image, held-out message, or new image-message combination evaluation.", "",
        "- Only one seed is represented, so seed variability and uncertainty intervals are unknown.", "",
        "- RGB1/RGB2 were validated sparsely, preventing fair cross-run comparison at RGB3-only epochs.", "",
        "- Checkpoint selection on this same fixed bank can favor bank-specific operating points.", "",
        "- The three tested loss weights do not identify where reconstruction gains saturate or reverse.", "",
        "## 11. Should We Test RGB=4 Next?", "",
        "RGB=4 would test whether the final-epoch RGB reconstruction trend continues beyond weight 3 and whether recovery metrics remain competitive under the same fixed-bank protocol.", "",
        "Reasons not to proceed immediately are that only one seed is available, the evaluation bank is not independent, and sparse RGB1/RGB2 validation limits trajectory-level comparison. A repeat-seed study and a held-out evaluation protocol would provide stronger evidence before attributing any RGB4 change to a stable loss-weight effect.", "",
        "The current analysis therefore supplies evidence for and against a next ablation point but does not make the RGB4 decision.", "",
        "### Generated plots", "",
        "- `rgb_psnr_vs_epoch.png`", "", "- `ber_vs_epoch.png`", "", "- `ber_vs_epoch_log.png`", "", "- `message_accuracy_vs_epoch.png`", "", "- `psnr_vs_ber.png`", "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "analysis_dir": str(OUT), "integrity": audit["status"],
        "final": selected_row(final), "best_psnr": selected_row(best_psnr),
        "best_ber": selected_row(best_ber), "best_ssim": selected_row(best_ssim),
        "zero_ber_epochs": [row["epoch"] for row in zero_ber],
        "exact_one_epochs": [row["epoch"] for row in exact_one],
        "common_epochs": common_epochs, "pearson": pearson, "spearman": spearman,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
