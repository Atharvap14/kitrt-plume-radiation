#!/usr/bin/env python3
"""Generate a structured axisymmetric plume radiation fixture CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--z-cells", type=int, default=32)
    parser.add_argument("--r-cells", type=int, default=32)
    parser.add_argument("--temperature-k", type=float, default=1200.0)
    parser.add_argument("--kappa-1-per-m", type=float, default=0.1)
    args = parser.parse_args()

    if args.z_cells < 2 or args.r_cells < 2:
        raise SystemExit("z-cells and r-cells must be at least 2")
    if args.z_max <= args.z_min:
        raise SystemExit("z-max must be larger than z-min")
    if args.r_max <= 0.0:
        raise SystemExit("r-max must be positive")
    if args.temperature_k < 0.0 or args.kappa_1_per_m < 0.0:
        raise SystemExit("temperature and kappa must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_m", "r_m", "temperature_K", "kappa_1_per_m"])
        for iz in range(args.z_cells):
            z = args.z_min + (args.z_max - args.z_min) * iz / (args.z_cells - 1)
            for ir in range(args.r_cells):
                r = args.r_max * ir / (args.r_cells - 1)
                writer.writerow([f"{z:.17g}", f"{r:.17g}", f"{args.temperature_k:.17g}", f"{args.kappa_1_per_m:.17g}"])


if __name__ == "__main__":
    main()
