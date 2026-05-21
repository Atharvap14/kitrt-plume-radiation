#!/usr/bin/env python3
"""Benchmark RadLib non-gray property models against shipped LBL GT curves.

The benchmark runs the same 1D parallel-plane setup for S1 and S2 using
Planck-mean, WSGG, and RCSLW with several spectral group counts. LBL data
shipped with RadLib are treated as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


METHOD_SPECS = [
    ("planckmean", 1, "Planck Mean"),
    ("wsgg", 4, "WSGG"),
    ("rcslw", 4, "RCSLW-4"),
    ("rcslw", 8, "RCSLW-8"),
    ("rcslw", 16, "RCSLW-16"),
    ("rcslw", 24, "RCSLW-24"),
    ("rcslw", 25, "RCSLW-25"),
]


@dataclass(frozen=True)
class BenchmarkRecord:
    case: str
    method: str
    label: str
    nGG: int
    elapsed_s: float
    max_rel_error: float
    mean_rel_error: float


def read_two_column_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.replace(",", " ").split()
            if len(fields) >= 2:
                xs.append(float(fields[0]))
                ys.append(float(fields[1]))
    if not xs:
        raise ValueError(f"No numeric rows found in {path}")
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def compile_driver(radlib_root: Path, source: Path, executable: Path) -> None:
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError("No C++ compiler found in PATH")

    include_dir = radlib_root / "installed" / "include"
    library_path = radlib_root / "installed" / "lib" / "libradlib.dylib"
    if not include_dir.exists() or not library_path.exists():
        raise FileNotFoundError(
            f"Missing RadLib installed headers/library under {radlib_root}. "
            "Run cmake --install for RadLib first."
        )

    executable.parent.mkdir(parents=True, exist_ok=True)
    command = [
        cxx,
        "-std=c++17",
        "-O3",
        str(source),
        f"-I{include_dir}",
        str(library_path),
        f"-Wl,-rpath,{library_path.parent}",
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True)


def parse_driver_output(stdout: str) -> Tuple[float, np.ndarray, np.ndarray]:
    elapsed_s: float | None = None
    xs: List[float] = []
    ys: List[float] = []
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if not fields:
            continue
        if fields[0] == "meta" and len(fields) == 5 and fields[1] != "case":
            elapsed_s = float(fields[4])
        elif fields[0] == "data" and len(fields) == 3 and fields[1] != "Lcold_m":
            xs.append(float(fields[1]))
            ys.append(float(fields[2]))
    if elapsed_s is None or not xs:
        raise ValueError(f"Could not parse benchmark driver output:\n{stdout}")
    return elapsed_s, np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def run_driver(executable: Path, case: str, method: str, nGG: int) -> Tuple[float, np.ndarray, np.ndarray]:
    result = subprocess.run(
        [str(executable), "--case", case, "--method", method, "--ngg", str(nGG)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_driver_output(result.stdout)


def write_summary(path: Path, records: Sequence[BenchmarkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "label", "method", "nGG", "elapsed_s", "max_rel_error", "mean_rel_error"])
        for row in records:
            writer.writerow(
                [
                    row.case,
                    row.label,
                    row.method,
                    row.nGG,
                    f"{row.elapsed_s:.9g}",
                    f"{row.max_rel_error:.9g}",
                    f"{row.mean_rel_error:.9g}",
                ]
            )


def write_points(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "label", "method", "nGG", "Lcold_m", "model_flux", "gt_flux", "rel_error"])
        writer.writerows(rows)


def _nice_label(value: float) -> str:
    if value >= 1.0:
        return f"{value:.2g}"
    if value >= 1.0e-2:
        return f"{value:.2f}"
    return f"{value:.1e}"


def write_grouped_bar_svg(
    path: Path,
    records: Sequence[BenchmarkRecord],
    metric: str,
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> None:
    cases = ["S1", "S2"]
    labels = [spec[2] for spec in METHOD_SPECS]
    lookup: Dict[Tuple[str, str], BenchmarkRecord] = {(r.case, r.label): r for r in records}
    values = np.array([[getattr(lookup[(case, label)], metric) for label in labels] for case in cases], dtype=np.float64)

    width = 1160
    height = 660
    left = 92
    right = 36
    top = 58
    bottom = 168
    plot_w = width - left - right
    plot_h = height - top - bottom

    if log_scale:
        floor = 1.0e-5
        values_for_scale = np.maximum(values, floor)
        ymin = 10.0 ** math.floor(math.log10(float(np.min(values_for_scale)) * 0.8))
        ymax = 10.0 ** math.ceil(math.log10(float(np.max(values_for_scale)) * 1.2))

        def sy(value: float) -> float:
            clipped = max(value, floor)
            return top + (math.log10(ymax) - math.log10(clipped)) / (math.log10(ymax) - math.log10(ymin)) * plot_h

        y_ticks = [10.0**p for p in range(int(math.log10(ymin)), int(math.log10(ymax)) + 1)]
    else:
        ymin = 0.0
        ymax = float(np.max(values) * 1.18)

        def sy(value: float) -> float:
            return top + (ymax - value) / (ymax - ymin) * plot_h

        y_ticks = np.linspace(ymin, ymax, 6)

    colors = {"S1": "#335c81", "S2": "#c75c2d"}
    group_w = plot_w / len(labels)
    bar_w = min(42.0, group_w * 0.34)
    bars: List[str] = []
    labels_svg: List[str] = []
    tick_svg: List[str] = []

    for tick in y_ticks:
        y = sy(float(tick))
        tick_svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        tick_svg.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end">{_nice_label(float(tick))}</text>')

    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        labels_svg.append(
            f'<text x="{cx:.2f}" y="{height - 108}" text-anchor="end" transform="rotate(-35 {cx:.2f},{height - 108})">{label}</text>'
        )
        for j, case in enumerate(cases):
            value = values[j, i]
            x = cx - bar_w + j * bar_w
            y = sy(float(value))
            h = top + plot_h - y
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w - 2:.2f}" height="{h:.2f}" '
                f'fill="{colors[case]}"/>'
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; font-size: 15px; }}
  .title {{ font-size: 24px; font-weight: 700; }}
  .axis {{ stroke: #1f2933; stroke-width: 1.6; }}
  .grid {{ stroke: #d5dde5; stroke-width: 1; }}
  .legend-box {{ fill: white; stroke: #b8c2cc; stroke-width: 1; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="36" class="title">{title}</text>
{''.join(tick_svg)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>
{''.join(bars)}
{''.join(labels_svg)}
<text transform="translate(26,{top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle">{ylabel}</text>
<rect x="{left + plot_w - 170}" y="{top + 18}" width="150" height="72" rx="4" class="legend-box"/>
<rect x="{left + plot_w - 150}" y="{top + 38}" width="20" height="14" fill="{colors['S1']}"/>
<text x="{left + plot_w - 120}" y="{top + 51}">S1</text>
<rect x="{left + plot_w - 150}" y="{top + 66}" width="20" height="14" fill="{colors['S2']}"/>
<text x="{left + plot_w - 120}" y="{top + 79}">S2</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radlib-root", default=os.environ.get("RADLIB_ROOT", "/tmp/radlib"))
    parser.add_argument("--driver", default="tests/result/radlib_flux_benchmark.x")
    parser.add_argument("--summary-csv", default="tests/result/radlib_method_benchmark_summary.csv")
    parser.add_argument("--points-csv", default="tests/result/radlib_method_benchmark_points.csv")
    parser.add_argument("--speed-svg", default="tests/result/radlib_method_speed.svg")
    parser.add_argument("--fidelity-svg", default="tests/result/radlib_method_fidelity.svg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    radlib_root = Path(args.radlib_root).expanduser().resolve()
    driver = (repo_root / args.driver).resolve()
    source = repo_root / "tools" / "cpp" / "radlib_flux_benchmark.cpp"
    compile_driver(radlib_root, source, driver)

    records: List[BenchmarkRecord] = []
    point_rows: List[Sequence[object]] = []
    for case in ["S1", "S2"]:
        gt_x, gt_y = read_two_column_data(radlib_root / "examples" / "python" / "LBLdata" / f"ex_{case}_LBL.dat")
        for method, nGG, label in METHOD_SPECS:
            elapsed_s, model_x, model_y = run_driver(driver, case, method, nGG)
            model_y_at_gt = np.interp(gt_x, model_x, model_y)
            rel_errors = np.abs(model_y_at_gt - gt_y) / np.maximum(np.abs(gt_y), 1.0e-300)
            records.append(
                BenchmarkRecord(
                    case=case,
                    method=method,
                    label=label,
                    nGG=nGG,
                    elapsed_s=elapsed_s,
                    max_rel_error=float(np.max(rel_errors)),
                    mean_rel_error=float(np.mean(rel_errors)),
                )
            )
            for x, model, gt, rel in zip(gt_x, model_y_at_gt, gt_y, rel_errors):
                point_rows.append([case, label, method, nGG, f"{x:.12g}", f"{model:.12g}", f"{gt:.12g}", f"{rel:.12g}"])

    write_summary(repo_root / args.summary_csv, records)
    write_points(repo_root / args.points_csv, point_rows)
    write_grouped_bar_svg(
        repo_root / args.speed_svg,
        records,
        metric="elapsed_s",
        title="RadLib Non-Gray Method Speed",
        ylabel="runtime for full curve (s)",
        log_scale=False,
    )
    write_grouped_bar_svg(
        repo_root / args.fidelity_svg,
        records,
        metric="max_rel_error",
        title="RadLib Non-Gray Method Fidelity vs LBL GT",
        ylabel="max relative flux error",
        log_scale=True,
    )

    print("case,label,elapsed_s,max_rel_error,mean_rel_error")
    for row in records:
        print(f"{row.case},{row.label},{row.elapsed_s:.6f},{row.max_rel_error:.9e},{row.mean_rel_error:.9e}")
    print(f"summary_csv,{args.summary_csv}")
    print(f"points_csv,{args.points_csv}")
    print(f"speed_svg,{args.speed_svg}")
    print(f"fidelity_svg,{args.fidelity_svg}")


if __name__ == "__main__":
    main()
