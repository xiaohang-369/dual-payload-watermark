"""Read-only comparison of the two completed joint watermark experiments.

The script never changes files under ``runs``.  Every invocation creates a new
timestamped directory under ``analysis_results`` containing CSV, JSON, Markdown,
and an SVG Pareto plot.  It uses only the Python standard library.
"""

from __future__ import annotations

import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS = (
    PROJECT_ROOT / "runs" / "joint_10x20_clean_v1_full",
    PROJECT_ROOT / "runs" / "joint_10x20_rgb2_v1",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_validations(path: Path) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if event.get("kind") == "validation":
                validations.append(event)
    if not validations:
        raise ValueError(f"No validation events found: {path}")
    required = {
        "epoch", "global_step", "rgb_psnr", "rgb_ssim", "ber",
        "bit_accuracy", "message_accuracy",
    }
    for event in validations:
        missing = required.difference(event)
        if missing:
            raise ValueError(
                f"Validation epoch {event.get('epoch')} in {path} is missing: "
                + ", ".join(sorted(missing))
            )
    return validations


def select_best(validations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    # Secondary keys make tie handling deterministic and prefer the stronger
    # trade-off point.  The final key selects the earliest epoch on a full tie.
    return {
        "lowest_BER": min(
            validations,
            key=lambda row: (
                row["ber"], -row["message_accuracy"], -row["rgb_psnr"], row["epoch"]
            ),
        ),
        "highest_Exact": min(
            validations,
            key=lambda row: (
                -row["message_accuracy"], row["ber"], -row["rgb_psnr"], row["epoch"]
            ),
        ),
        "highest_RGB_PSNR": min(
            validations,
            key=lambda row: (
                -row["rgb_psnr"], row["ber"], -row["message_accuracy"], row["epoch"]
            ),
        ),
    }


def new_output_directory() -> Path:
    parent = PROJECT_ROOT / "analysis_results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = parent / f"joint_10x20_comparison_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for point in sorted(points, key=lambda row: (row["ber"], -row["rgb_psnr"])):
        dominated = any(
            other["ber"] <= point["ber"]
            and other["rgb_psnr"] >= point["rgb_psnr"]
            and (
                other["ber"] < point["ber"]
                or other["rgb_psnr"] > point["rgb_psnr"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return frontier


def write_pareto_svg(path: Path, series: list[dict[str, Any]]) -> None:
    width, height = 1000, 680
    left, right, top, bottom = 100, 45, 70, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for item in series for point in item["validations"]]
    x_values = [point["ber"] for point in all_points]
    y_values = [point["rgb_psnr"] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = (x_max - x_min) * 0.06 or 0.01
    y_pad = (y_max - y_min) * 0.08 or 1.0
    x_min = max(0.0, x_min - x_pad)
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ("#2563eb", "#dc2626")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#1f2937}.tick{font-size:12px}.label{font-size:15px}.title{font-size:22px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">BER vs RGB PSNR validation Pareto relationship</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = px(x_value)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        svg.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{x_value:.3f}</text>')
        y_value = y_min + fraction * (y_max - y_min)
        y = py(y_value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        svg.append(f'<text class="tick" x="{left - 14}" y="{y + 4:.2f}" text-anchor="end">{y_value:.1f}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
        f'<text class="label" x="{left + plot_width / 2}" y="{height - 30}" text-anchor="middle">BER (lower is better)</text>',
        f'<text class="label" x="25" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 25 {top + plot_height / 2})">RGB PSNR, dB (higher is better)</text>',
    ])

    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        ordered = sorted(item["validations"], key=lambda row: row["epoch"])
        coords = " ".join(f'{px(row["ber"]):.2f},{py(row["rgb_psnr"]):.2f}' for row in ordered)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="0.65"/>')
        for point in ordered:
            x, y = px(point["ber"]), py(point["rgb_psnr"])
            tooltip = escape(
                f'{item["experiment"]}: epoch {point["epoch"]}, '
                f'BER={point["ber"]:.8f}, RGB PSNR={point["rgb_psnr"]:.6f}'
            )
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="1"><title>{tooltip}</title></circle>')
        final = item["final"]
        x, y = px(final["ber"]), py(final["rgb_psnr"])
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="none" stroke="{color}" stroke-width="3"/>')

    frontier = pareto_frontier(all_points)
    frontier_coords = " ".join(
        f'{px(point["ber"]):.2f},{py(point["rgb_psnr"]):.2f}' for point in frontier
    )
    if frontier_coords:
        svg.append(f'<polyline points="{frontier_coords}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>')

    legend_x, legend_y = left + 18, top + 22
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        y = legend_y + index * 24
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text class="legend" x="{legend_x + 38}" y="{y + 4}">{escape(item["experiment"])}</text>')
    y = legend_y + len(series) * 24
    svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>')
    svg.append(f'<text class="legend" x="{legend_x + 38}" y="{y + 4}">non-dominated frontier</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    series = []
    comparison_rows = []
    best_rows = []
    for run_dir in RUNS:
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        config = read_json(config_path)
        validations = read_validations(metrics_path)
        final = max(validations, key=lambda row: (row["epoch"], row["global_step"]))
        experiment = config.get("experiment", {}).get("mode", run_dir.name)
        rgb_weight = config["loss"]["rgb"]
        item = {
            "experiment": experiment,
            "run_directory": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "rgb_loss": rgb_weight,
            "final": final,
            "validations": validations,
        }
        series.append(item)
        comparison_rows.append({
            "experiment": experiment,
            "rgb_loss": rgb_weight,
            "final_rgb_psnr": final["rgb_psnr"],
            "final_rgb_ssim": final["rgb_ssim"],
            "final_BER": final["ber"],
            # Per-bit message accuracy; it is exactly 1 - BER in these logs.
            "final_message_accuracy": final["bit_accuracy"],
            # Whole-message exact success; logged under message_accuracy.
            "final_exact_accuracy": final["message_accuracy"],
        })
        for criterion, event in select_best(validations).items():
            value_key = {
                "lowest_BER": "ber",
                "highest_Exact": "message_accuracy",
                "highest_RGB_PSNR": "rgb_psnr",
            }[criterion]
            best_rows.append({
                "experiment": experiment,
                "criterion": criterion,
                "epoch": event["epoch"],
                "global_step": event["global_step"],
                "value": event[value_key],
                "BER": event["ber"],
                "exact_accuracy": event["message_accuracy"],
                "rgb_psnr": event["rgb_psnr"],
            })

    output_dir = new_output_directory()
    comparison_headers = list(comparison_rows[0])
    with (output_dir / "experiment_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_headers)
        writer.writeheader()
        writer.writerows(comparison_rows)
    best_headers = list(best_rows[0])
    with (output_dir / "best_validation_checkpoints.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=best_headers)
        writer.writeheader()
        writer.writerows(best_rows)
    summary = {
        "metric_mapping": {
            "final_message_accuracy": "metrics.jsonl: bit_accuracy (1 - BER)",
            "final_exact_accuracy": "metrics.jsonl: message_accuracy (all 64 bits correct)",
        },
        "experiments": comparison_rows,
        "best_validation_checkpoints": best_rows,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_pareto_svg(output_dir / "pareto_ber_vs_rgb_psnr.svg", series)

    comparison_md = [[
        row["experiment"], row["rgb_loss"], f'{row["final_rgb_psnr"]:.6f}',
        f'{row["final_rgb_ssim"]:.6f}', f'{row["final_BER"]:.8f}',
        f'{row["final_message_accuracy"]:.6f}', f'{row["final_exact_accuracy"]:.6f}',
    ] for row in comparison_rows]
    best_md = [[
        row["experiment"], row["criterion"], row["epoch"], row["global_step"],
        f'{row["value"]:.8f}', f'{row["BER"]:.8f}',
        f'{row["exact_accuracy"]:.6f}', f'{row["rgb_psnr"]:.6f}',
    ] for row in best_rows]
    report = "\n".join([
        "# Joint 10x20 Experiment Comparison",
        "",
        "## Final Validation Comparison",
        "",
        markdown_table(comparison_headers, comparison_md),
        "",
        "`final_message_accuracy` is the logged per-bit `bit_accuracy` (1 - BER). "
        "`final_exact_accuracy` is the logged whole-message `message_accuracy`.",
        "",
        "## Best Validation Checkpoints",
        "",
        markdown_table(best_headers, best_md),
        "",
        "Tie-breaking is deterministic: the primary criterion is followed by the other "
        "quality metrics, then the earliest epoch.",
        "",
        "## Pareto Plot",
        "",
        "![BER versus RGB PSNR Pareto plot](pareto_ber_vs_rgb_psnr.svg)",
        "",
    ])
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Analysis directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
