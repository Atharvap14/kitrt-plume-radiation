#!/usr/bin/env python3
"""Validate a RadLib non-gray flux example against its LBL reference data.

This script expects an external RadLib checkout/build. It runs RadLib's C++
RCSLW examples, parses the shipped line-by-line reference data, writes a
comparison CSV, and creates an SVG plot of ground truth versus model output.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


CASE_TITLES = {
    "ex_S1": "RadLib Ex S1 Non-Gray Flux Validation",
    "ex_S2": "RadLib Ex S2 Non-Gray Flux Validation",
}


def _read_two_column_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.replace(",", " ").split()
            if len(fields) < 2:
                continue
            xs.append(float(fields[0]))
            ys.append(float(fields[1]))
    if not xs:
        raise ValueError(f"No numeric rows found in {path}")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _run_radlib_example(radlib_root: Path, case: str, executable: Path | None) -> Tuple[np.ndarray, np.ndarray, str]:
    exe = executable
    if exe is None:
        candidates = [
            radlib_root / "examples" / "c++" / f"{case}.x",
            radlib_root / "build" / "examples" / "c++" / f"{case}.x",
        ]
        exe = next((candidate for candidate in candidates if candidate.exists()), None)
    if exe is None or not exe.exists():
        raise FileNotFoundError(f"Could not find RadLib {case}.x. Build/install RadLib C++ examples first.")

    result = subprocess.run(
        [str(exe)],
        cwd=str(radlib_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    xs: List[float] = []
    ys: List[float] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) >= 2:
            xs.append(float(fields[0]))
            ys.append(float(fields[1]))
    if not xs:
        raise ValueError(f"RadLib example produced no parseable data:\n{result.stdout}")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), result.stdout


def _write_comparison_csv(
    path: Path,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    model_y_at_gt: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Lcold_m", "gt_lbl_flux", "model_rcslw_flux", "abs_error", "rel_error"])
        for x, gt, model in zip(gt_x, gt_y, model_y_at_gt):
            abs_error = model - gt
            rel_error = abs(abs_error) / max(abs(gt), 1.0e-300)
            writer.writerow([f"{x:.12g}", f"{gt:.12g}", f"{model:.12g}", f"{abs_error:.12g}", f"{rel_error:.12g}"])


def _polyline(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_svg_plot(
    path: Path,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    model_x: np.ndarray,
    model_y: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 620
    left = 88
    right = 36
    top = 52
    bottom = 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = np.concatenate([gt_x, model_x])
    all_y = np.concatenate([gt_y, model_y])
    xmin = float(np.min(all_x))
    xmax = float(np.max(all_x))
    ymin = float(np.min(all_y))
    ymax = float(np.max(all_y))
    ypad = 0.08 * max(ymax - ymin, 1.0e-9)
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    gt_points = [(sx(float(x)), sy(float(y))) for x, y in zip(gt_x, gt_y)]
    model_points = [(sx(float(x)), sy(float(y))) for x, y in zip(model_x, model_y)]

    x_ticks = np.linspace(xmin, xmax, 6)
    y_ticks = np.linspace(ymin, ymax, 6)
    tick_lines: List[str] = []
    for x in x_ticks:
        px = sx(float(x))
        tick_lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" class="grid"/>')
        tick_lines.append(f'<text x="{px:.2f}" y="{height - 42}" text-anchor="middle">{x:.2g}</text>')
    for y in y_ticks:
        py = sy(float(y))
        tick_lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" class="grid"/>')
        tick_lines.append(f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end">{y:.3g}</text>')

    gt_circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.0" class="gt-point"/>' for x, y in gt_points
    )
    model_circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" class="model-point"/>' for x, y in model_points
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; font-size: 16px; }}
  .title {{ font-size: 24px; font-weight: 700; }}
  .axis {{ stroke: #1f2933; stroke-width: 1.6; }}
  .grid {{ stroke: #d5dde5; stroke-width: 1; }}
  .gt-line {{ fill: none; stroke: #0b6e4f; stroke-width: 3.2; }}
  .model-line {{ fill: none; stroke: #b83232; stroke-width: 3.2; }}
  .gt-point {{ fill: #0b6e4f; stroke: white; stroke-width: 1.4; }}
  .model-point {{ fill: #b83232; stroke: white; stroke-width: 1.2; }}
  .legend-box {{ fill: white; stroke: #b8c2cc; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="34" class="title">{html.escape(title)}</text>
{''.join(tick_lines)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
<polyline points="{_polyline(gt_points)}" class="gt-line"/>
<polyline points="{_polyline(model_points)}" class="model-line"/>
{gt_circles}
{model_circles}
<text x="{left + plot_w / 2:.2f}" y="{height - 12}" text-anchor="middle">Lcold (m)</text>
<text transform="translate(24,{top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">q(L) / sigma Thot^4</text>
<rect x="{left + plot_w - 286}" y="{top + 16}" width="270" height="72" rx="4" class="legend-box"/>
<line x1="{left + plot_w - 266}" y1="{top + 40}" x2="{left + plot_w - 220}" y2="{top + 40}" class="gt-line"/>
<text x="{left + plot_w - 208}" y="{top + 45}">GT: LBL reference</text>
<line x1="{left + plot_w - 266}" y1="{top + 68}" x2="{left + plot_w - 220}" y2="{top + 68}" class="model-line"/>
<text x="{left + plot_w - 208}" y="{top + 73}">Model: RadLib RCSLW</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_TITLES), default="ex_S2")
    parser.add_argument("--radlib-root", default=os.environ.get("RADLIB_ROOT", "/tmp/radlib"))
    parser.add_argument("--executable", default=None, help="Optional path to a RadLib example executable")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-svg", default=None)
    parser.add_argument("--max-rel-error", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radlib_root = Path(args.radlib_root).expanduser().resolve()
    executable = Path(args.executable).expanduser().resolve() if args.executable else None
    output_csv = args.output_csv or f"tests/result/radlib_{args.case}_flux_validation.csv"
    plot_svg = args.plot_svg or f"tests/result/radlib_{args.case}_flux_validation.svg"
    lbl_path = radlib_root / "examples" / "python" / "LBLdata" / f"{args.case}_LBL.dat"
    if not lbl_path.exists():
        raise FileNotFoundError(f"Missing RadLib LBL reference: {lbl_path}")

    model_x, model_y, _ = _run_radlib_example(radlib_root, args.case, executable)
    gt_x, gt_y = _read_two_column_data(lbl_path)
    model_y_at_gt = np.interp(gt_x, model_x, model_y)
    rel_errors = np.abs(model_y_at_gt - gt_y) / np.maximum(np.abs(gt_y), 1.0e-300)

    _write_comparison_csv(Path(output_csv), gt_x, gt_y, model_y_at_gt)
    _write_svg_plot(
        Path(plot_svg),
        gt_x,
        gt_y,
        model_x,
        model_y,
        CASE_TITLES[args.case],
    )

    print("case,model,gt_points,max_rel_error,mean_rel_error,output_csv,plot_svg")
    print(
        f"RadLib {args.case},RCSLW-24 vs shipped LBL,"
        f"{len(gt_x)},{float(np.max(rel_errors)):.9e},{float(np.mean(rel_errors)):.9e},"
        f"{output_csv},{plot_svg}"
    )
    if float(np.max(rel_errors)) > args.max_rel_error:
        raise SystemExit(f"Max relative error exceeded {args.max_rel_error:g}")


if __name__ == "__main__":
    main()
