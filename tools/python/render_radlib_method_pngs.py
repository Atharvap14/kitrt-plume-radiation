#!/usr/bin/env python3
"""Render readable PNG plots from RadLib method benchmark CSV outputs.

The SVGs are useful for vector output, but macOS Quick Look thumbnails can crop
wide SVG content. This renderer draws PNGs directly with Pillow so labels and
legends remain visible in chat previews and reports.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


INK = "#1f2933"
GRID = "#d5dde5"
S1_COLOR = "#335c81"
S2_COLOR = "#c75c2d"
GT_COLOR = "#111827"

METHOD_COLORS = {
    "WSGG": "#f28c28",
    "RCSLW-4": "#6a994e",
    "RCSLW-8": "#0077b6",
    "RCSLW-16": "#7b2cbf",
    "RCSLW-24": "#d00000",
    "RCSLW-25": "#4d908e",
    "RCSLW-local-8": "#00a896",
    "RCSLW-local-16": "#ef476f",
    "RCSLW-local-24": "#073b4c",
}


@dataclass(frozen=True)
class SummaryRow:
    case: str
    label: str
    elapsed_s: float
    max_rel_error: float
    mean_rel_error: float


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return int(box[2] - box[0]), int(box[3] - box[1])


def read_summary(path: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SummaryRow(
                    case=row["case"],
                    label=row["label"],
                    elapsed_s=float(row["elapsed_s"]),
                    max_rel_error=float(row["max_rel_error"]),
                    mean_rel_error=float(row["mean_rel_error"]),
                )
            )
    return rows


def read_points(path: Path) -> Dict[str, Dict[str, List[Tuple[float, float, float]]]]:
    points: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points[row["case"]][row["label"]].append(
                (float(row["Lcold_m"]), float(row["model_flux"]), float(row["gt_flux"]))
            )
    for case_rows in points.values():
        for method_rows in case_rows.values():
            method_rows.sort(key=lambda item: item[0])
    return points


def nice_number(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 100 or abs(value) < 0.01:
        return f"{value:.1e}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def draw_legend(draw: ImageDraw.ImageDraw, items: Sequence[Tuple[str, str]], x: int, y: int) -> None:
    label_font = font(30)
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 46
        draw.rectangle([x, yy, x + 34, yy + 22], fill=color)
        draw.text((x + 50, yy - 4), label, font=label_font, fill=INK)


def render_bar_png(
    path: Path,
    rows: Sequence[SummaryRow],
    metric: str,
    title: str,
    x_label: str,
    log_scale: bool = False,
) -> None:
    methods: List[str] = []
    for row in rows:
        if row.label not in methods:
            methods.append(row.label)
    lookup = {(row.case, row.label): row for row in rows}

    width = 1900
    row_h = 82
    height = 260 + row_h * len(methods)
    left = 360
    right = 300
    top = 170
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(56, bold=True)
    label_font = font(32)
    tick_font = font(28)
    small_font = font(24)

    draw.text((left, 44), title, font=title_font, fill=INK)
    draw_legend(draw, [("S1", S1_COLOR), ("S2", S2_COLOR)], width - right + 50, 50)

    values = [getattr(row, metric) for row in rows]
    if log_scale:
        xmin = 1.0e-4
        xmax = 10.0

        def sx(value: float) -> float:
            clipped = max(value, xmin)
            return left + (math.log10(clipped) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * plot_w

        ticks = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]
    else:
        xmin = 0.0
        xmax = max(values) * 1.18
        step = 1.0 if xmax > 3.0 else 0.5
        xmax = math.ceil(xmax / step) * step

        def sx(value: float) -> float:
            return left + (value - xmin) / (xmax - xmin) * plot_w

        ticks = [i * step for i in range(int(round(xmax / step)) + 1)]

    for tick in ticks:
        x = sx(tick)
        draw.line([(x, top), (x, top + plot_h)], fill=GRID, width=2)
        label = nice_number(tick)
        tw, _ = text_size(draw, label, tick_font)
        draw.text((x - tw / 2, top + plot_h + 24), label, font=tick_font, fill=INK)

    draw.line([(left, top), (left, top + plot_h)], fill=INK, width=3)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=INK, width=3)

    bar_h = 24
    for idx, method in enumerate(methods):
        y_center = top + idx * row_h + row_h / 2
        draw.text((48, y_center - 22), method, font=label_font, fill=INK)
        for case, color, offset in [("S1", S1_COLOR, -15), ("S2", S2_COLOR, 15)]:
            row = lookup[(case, method)]
            value = getattr(row, metric)
            x0 = sx(xmin if log_scale else 0.0)
            x1 = sx(value)
            y0 = y_center + offset - bar_h / 2
            y1 = y_center + offset + bar_h / 2
            draw.rectangle([x0, y0, x1, y1], fill=color)
            draw.text((min(x1 + 10, left + plot_w - 100), y0 - 5), nice_number(value), font=small_font, fill=INK)

    tw, _ = text_size(draw, x_label, label_font)
    draw.text((left + plot_w / 2 - tw / 2, height - 54), x_label, font=label_font, fill=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_curve_png(path: Path, case: str, points: Dict[str, List[Tuple[float, float, float]]]) -> None:
    width = 2100
    height = 1250
    left = 150
    right = 560
    top = 150
    bottom = 150
    plot_w = width - left - right
    plot_h = height - top - bottom

    methods = [label for label in METHOD_COLORS if label in points]
    gt_rows = next(iter(points.values()))
    gt_x = [row[0] for row in gt_rows]
    gt_y = [row[2] for row in gt_rows]

    model_values = [value for label in methods for _, value, _ in points[label]]
    all_x = gt_x + [x for label in methods for x, _, _ in points[label]]
    all_y = gt_y + model_values
    xmin = min(all_x)
    xmax = max(all_x)
    ymin = min(all_y)
    ymax = max(all_y)
    ypad = 0.08 * max(ymax - ymin, 1.0e-9)
    ymin -= ypad
    ymax += ypad

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(56, bold=True)
    label_font = font(32)
    tick_font = font(28)

    draw.text((left, 44), f"RadLib {case}: GT vs Non-Gray Models", font=title_font, fill=INK)
    draw.text((left, 106), "Planck Mean is omitted from this curve plot because it is off-scale; see fidelity bars.", font=font(26), fill="#5b6673")

    for tick in [xmin + (xmax - xmin) * i / 5.0 for i in range(6)]:
        x = sx(tick)
        draw.line([(x, top), (x, top + plot_h)], fill=GRID, width=2)
        label = nice_number(tick)
        tw, _ = text_size(draw, label, tick_font)
        draw.text((x - tw / 2, top + plot_h + 24), label, font=tick_font, fill=INK)

    for tick in [ymin + (ymax - ymin) * i / 5.0 for i in range(6)]:
        y = sy(tick)
        draw.line([(left, y), (left + plot_w, y)], fill=GRID, width=2)
        label = nice_number(tick)
        tw, th = text_size(draw, label, tick_font)
        draw.text((left - tw - 16, y - th / 2), label, font=tick_font, fill=INK)

    draw.line([(left, top), (left, top + plot_h)], fill=INK, width=3)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=INK, width=3)

    gt_points = [(sx(x), sy(y)) for x, y in zip(gt_x, gt_y)]
    draw.line(gt_points, fill=GT_COLOR, width=7, joint="curve")
    for x, y in gt_points:
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=GT_COLOR, outline="white", width=3)

    for label in methods:
        rows = points[label]
        curve = [(sx(x), sy(model_y)) for x, model_y, _ in rows]
        color = METHOD_COLORS[label]
        draw.line(curve, fill=color, width=5, joint="curve")
        for x, y in curve:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)

    x_axis = "Lcold (m)"
    tw, _ = text_size(draw, x_axis, label_font)
    draw.text((left + plot_w / 2 - tw / 2, height - 62), x_axis, font=label_font, fill=INK)
    y_axis = "normalized final flux"
    y_img = Image.new("RGBA", (420, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_img)
    y_draw.text((0, 0), y_axis, font=label_font, fill=INK)
    y_img = y_img.rotate(90, expand=True)
    image.paste(y_img, (32, top + plot_h // 2 - y_img.height // 2), y_img)

    legend_x = left + plot_w + 52
    legend_y = top + 20
    legend_items = [("GT LBL", GT_COLOR)] + [(label, METHOD_COLORS[label]) for label in methods]
    draw_legend(draw, legend_items, legend_x, legend_y)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", default="tests/result/radlib_method_benchmark_summary.csv")
    parser.add_argument("--points-csv", default="tests/result/radlib_method_benchmark_points.csv")
    parser.add_argument("--output-dir", default="tests/result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    points_path = Path(args.points_csv)
    output_dir = Path(args.output_dir)
    rows = read_summary(summary_path)
    points = read_points(points_path)

    render_bar_png(
        output_dir / "radlib_method_speed_readable.png",
        rows,
        "elapsed_s",
        "RadLib Non-Gray Method Speed",
        "runtime for full S1/S2 curve (s)",
        log_scale=False,
    )
    render_bar_png(
        output_dir / "radlib_method_fidelity_readable.png",
        rows,
        "max_rel_error",
        "RadLib Non-Gray Method Fidelity vs LBL GT",
        "max relative flux error",
        log_scale=True,
    )
    render_curve_png(output_dir / "radlib_S1_gt_vs_models_readable.png", "S1", points["S1"])
    render_curve_png(output_dir / "radlib_S2_gt_vs_models_readable.png", "S2", points["S2"])


if __name__ == "__main__":
    main()
