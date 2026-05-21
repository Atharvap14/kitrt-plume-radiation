#!/usr/bin/env python3
"""Generate small RADIS line-by-line gas-cell fixtures.

RADIS/HITEMP line-by-line spectra are the highest-fidelity validation target
for plume-relevant gases, but full plume-sized LBL runs are too expensive for
routine development. This utility creates deliberately small homogeneous
gas-cell fixtures, then compares the LBL result with simple Planck-weighted
band-gray reductions. The output is meant to screen non-gray property handling
before wiring a model into the plume solver.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GasCellCase:
    name: str
    species: str
    isotope: str
    wmin_cm_1: float
    wmax_cm_1: float
    wstep_cm_1: float
    temperature_K: float
    pressure_bar: float
    mole_fraction: float
    path_length_cm: float
    note: str


CASES: Dict[str, GasCellCase] = {
    "co2_hot_2300": GasCellCase(
        name="co2_hot_2300",
        species="CO2",
        isotope="1",
        wmin_cm_1=2300.0,
        wmax_cm_1=2305.0,
        wstep_cm_1=0.01,
        temperature_K=1800.0,
        pressure_bar=1.01325,
        mole_fraction=0.10,
        path_length_cm=1.0,
        note="Hot CO2 band slice near 4.35 um.",
    ),
    "co_hot_2100": GasCellCase(
        name="co_hot_2100",
        species="CO",
        isotope="1",
        wmin_cm_1=2100.0,
        wmax_cm_1=2105.0,
        wstep_cm_1=0.01,
        temperature_K=1800.0,
        pressure_bar=1.01325,
        mole_fraction=0.10,
        path_length_cm=1.0,
        note="Hot CO fundamental-band slice near 4.76 um.",
    ),
    "h2o_hot_1500": GasCellCase(
        name="h2o_hot_1500",
        species="H2O",
        isotope="1",
        wmin_cm_1=1500.0,
        wmax_cm_1=1501.0,
        wstep_cm_1=0.01,
        temperature_K=1800.0,
        pressure_bar=1.01325,
        mole_fraction=0.10,
        path_length_cm=1.0,
        note="Small hot H2O screening slice; first database fetch can be slow.",
    ),
}


@contextmanager
def time_limit(seconds: int) -> Iterable[None]:
    if seconds <= 0:
        yield
        return

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"RADIS case exceeded --timeout-s={seconds}")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def parse_groups(raw: str) -> List[int]:
    groups: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("Group counts must be positive")
        groups.append(value)
    return sorted(set(groups))


def planck_radiance_wavenumber(wn_cm_1: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return blackbody radiance in mW/cm2/sr/cm^-1."""
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    sigma_m_1 = wn_cm_1 * 100.0
    exponent = h * c * sigma_m_1 / (k * temperature_K)
    per_m_1 = 2.0 * h * c * c * sigma_m_1**3 / np.expm1(exponent)
    return per_m_1 * 10.0


def integrated_flux_w_m2(wn_cm_1: np.ndarray, radiance_mw_cm2_sr_cm: np.ndarray) -> float:
    """Assume hemispherical diffuse emission: q = pi * integral(I_nu dnu)."""
    band_radiance = float(np.trapezoid(radiance_mw_cm2_sr_cm, wn_cm_1))
    return math.pi * band_radiance * 10.0


def build_group_model(
    wn_cm_1: np.ndarray,
    abscoeff_cm_1: np.ndarray,
    temperature_K: float,
    path_length_cm: float,
    groups: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blackbody = planck_radiance_wavenumber(wn_cm_1, temperature_K)
    edges = np.linspace(float(wn_cm_1[0]), float(wn_cm_1[-1]), groups + 1)
    kappa_group = np.zeros_like(wn_cm_1)

    for idx in range(groups):
        if idx == groups - 1:
            mask = (wn_cm_1 >= edges[idx]) & (wn_cm_1 <= edges[idx + 1])
        else:
            mask = (wn_cm_1 >= edges[idx]) & (wn_cm_1 < edges[idx + 1])
        if not np.any(mask):
            continue
        denom = float(np.trapezoid(blackbody[mask], wn_cm_1[mask]))
        if abs(denom) <= 1.0e-300:
            kappa = float(np.mean(abscoeff_cm_1[mask]))
        else:
            kappa = float(np.trapezoid(abscoeff_cm_1[mask] * blackbody[mask], wn_cm_1[mask]) / denom)
        kappa_group[mask] = max(kappa, 0.0)

    transmittance = np.exp(-kappa_group * path_length_cm)
    radiance = blackbody * (1.0 - transmittance)
    return kappa_group, transmittance, radiance


def require_radis() -> object:
    try:
        from radis import calc_spectrum
    except ImportError as exc:
        raise SystemExit(
            "RADIS is not installed in this Python environment. Install it with "
            "`python -m pip install radis`, or run through the `kitrt-radis` conda env."
        ) from exc
    return calc_spectrum


def check_hitemp_credentials(args: argparse.Namespace) -> None:
    if args.databank.lower() != "hitemp":
        return
    has_env_creds = bool(os.environ.get("HITRAN_EMAIL") and os.environ.get("HITRAN_PASSWORD"))
    if has_env_creds or args.allow_stored_hitran_credentials:
        return
    raise SystemExit(
        "RADIS HITEMP downloads require HITRAN credentials. Set HITRAN_EMAIL and "
        "HITRAN_PASSWORD, or pass --allow-stored-hitran-credentials if RADIS already "
        "has encrypted credentials in its local config. Use --databank hitran only "
        "for low-temperature smoke tests; it is not high-temperature GT."
    )


def run_radis_case(case: GasCellCase, args: argparse.Namespace) -> Tuple[object, float]:
    calc_spectrum = require_radis()
    start = time.perf_counter()
    with time_limit(args.timeout_s):
        spectrum = calc_spectrum(
            wmin=case.wmin_cm_1,
            wmax=case.wmax_cm_1,
            species=case.species,
            isotope=case.isotope,
            pressure=case.pressure_bar,
            Tgas=case.temperature_K,
            mole_fraction=case.mole_fraction,
            path_length=case.path_length_cm,
            databank=args.databank,
            wstep=case.wstep_cm_1,
            verbose=args.verbose,
        )
    return spectrum, time.perf_counter() - start


def write_outputs(
    case: GasCellCase,
    databank: str,
    elapsed_s: float,
    groups: Sequence[int],
    spectrum: object,
    out_dir: Path,
) -> Tuple[Path, Path, Path]:
    wn, trans_lbl = spectrum.get("transmittance_noslit")
    _, rad_lbl = spectrum.get("radiance_noslit")
    _, abscoeff = spectrum.get("abscoeff")
    wn = np.asarray(wn, dtype=float)
    trans_lbl = np.asarray(trans_lbl, dtype=float)
    rad_lbl = np.asarray(rad_lbl, dtype=float)
    abscoeff = np.asarray(abscoeff, dtype=float)

    models = {
        count: build_group_model(wn, abscoeff, case.temperature_K, case.path_length_cm, count)
        for count in groups
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"radis_{case.name}_{databank.lower()}"
    spectral_csv = prefix.with_name(prefix.name + "_spectra.csv")
    summary_csv = prefix.with_name(prefix.name + "_summary.csv")
    plot_png = prefix.with_name(prefix.name + "_gt_vs_group_models.png")

    with spectral_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "databank",
                "model",
                "groups",
                "wavenumber_cm_1",
                "abscoeff_cm_1",
                "transmittance",
                "radiance_mW_cm2_sr_cm_1",
            ]
        )
        for idx, value in enumerate(wn):
            writer.writerow([case.name, databank, "RADIS-LBL-GT", 0, value, abscoeff[idx], trans_lbl[idx], rad_lbl[idx]])
        for count, (_, trans_model, rad_model) in models.items():
            for idx, value in enumerate(wn):
                writer.writerow([case.name, databank, "Planck-band-gray", count, value, "", trans_model[idx], rad_model[idx]])

    gt_flux = integrated_flux_w_m2(wn, rad_lbl)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "species",
                "databank",
                "model",
                "groups",
                "elapsed_s",
                "spectral_points",
                "temperature_K",
                "pressure_bar",
                "mole_fraction",
                "path_length_cm",
                "band_radiance_mW_cm2_sr",
                "hemispherical_flux_W_m2",
                "flux_rel_error",
                "mean_transmittance",
                "max_abscoeff_cm_1",
            ]
        )
        writer.writerow(
            [
                case.name,
                case.species,
                databank,
                "RADIS-LBL-GT",
                0,
                f"{elapsed_s:.6g}",
                len(wn),
                case.temperature_K,
                case.pressure_bar,
                case.mole_fraction,
                case.path_length_cm,
                f"{float(np.trapezoid(rad_lbl, wn)):.12g}",
                f"{gt_flux:.12g}",
                "0",
                f"{float(np.mean(trans_lbl)):.12g}",
                f"{float(np.max(abscoeff)):.12g}",
            ]
        )
        for count, (kappa_group, trans_model, rad_model) in models.items():
            model_flux = integrated_flux_w_m2(wn, rad_model)
            rel_error = abs(model_flux - gt_flux) / max(abs(gt_flux), 1.0e-300)
            writer.writerow(
                [
                    case.name,
                    case.species,
                    databank,
                    "Planck-band-gray",
                    count,
                    f"{elapsed_s:.6g}",
                    len(wn),
                    case.temperature_K,
                    case.pressure_bar,
                    case.mole_fraction,
                    case.path_length_cm,
                    f"{float(np.trapezoid(rad_model, wn)):.12g}",
                    f"{model_flux:.12g}",
                    f"{rel_error:.12g}",
                    f"{float(np.mean(trans_model)):.12g}",
                    f"{float(np.max(kappa_group)):.12g}",
                ]
            )

    render_plot(plot_png, case, databank, wn, trans_lbl, rad_lbl, models, gt_flux)
    return spectral_csv, summary_csv, plot_png


def render_plot(
    path: Path,
    case: GasCellCase,
    databank: str,
    wn: np.ndarray,
    trans_lbl: np.ndarray,
    rad_lbl: np.ndarray,
    models: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gt_flux: float,
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#f0e442", "#56b4e9"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(wn, rad_lbl, color="#111827", linewidth=2.2, label="GT RADIS LBL")
    axes[1].plot(wn, trans_lbl, color="#111827", linewidth=2.2, label="GT RADIS LBL")

    for idx, (count, (_, trans_model, rad_model)) in enumerate(models.items()):
        color = colors[idx % len(colors)]
        model_flux = integrated_flux_w_m2(wn, rad_model)
        rel_error = abs(model_flux - gt_flux) / max(abs(gt_flux), 1.0e-300)
        label = f"Planck-band-gray-{count} (flux err {rel_error:.2e})"
        axes[0].plot(wn, rad_model, color=color, linewidth=1.7, linestyle="--", label=label)
        axes[1].plot(wn, trans_model, color=color, linewidth=1.7, linestyle="--", label=f"Planck-band-gray-{count}")

    title = f"RADIS {databank.upper()} {case.species} Gas Cell: GT vs Group Models"
    subtitle = (
        f"{case.temperature_K:.0f} K, {case.pressure_bar:.4g} bar, "
        f"x={case.mole_fraction:.3g}, L={case.path_length_cm:.3g} cm, "
        f"{case.wmin_cm_1:.2f}-{case.wmax_cm_1:.2f} cm^-1"
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=12)

    axes[0].set_ylabel("Radiance (mW/cm2/sr/cm^-1)")
    axes[1].set_ylabel("Transmittance")
    axes[1].set_xlabel("Wavenumber (cm^-1)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=10, frameon=True)
    axes[1].legend(loc="best", fontsize=10, frameon=True)
    axes[0].text(
        0.01,
        0.94,
        f"GT hemispherical band flux: {gt_flux:.6g} W/m2",
        transform=axes[0].transAxes,
        fontsize=11,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    fig.tight_layout(rect=[0.03, 0.03, 0.995, 0.925])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def selected_cases(raw: str) -> List[GasCellCase]:
    if raw == "all":
        return list(CASES.values())
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Available: {', '.join(CASES)}")
    return [CASES[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="co2_hot_2300", help="Case name, comma-separated names, or 'all'")
    parser.add_argument("--databank", default="hitemp", choices=["hitemp", "hitran"], help="RADIS line database")
    parser.add_argument("--groups", default="1,4,8,16", help="Comma-separated band-gray group counts")
    parser.add_argument("--out-dir", type=Path, default=Path("tests/result"), help="Output directory")
    parser.add_argument("--timeout-s", type=int, default=180, help="Per-case RADIS timeout; <=0 disables")
    parser.add_argument("--allow-stored-hitran-credentials", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable RADIS verbose output")
    args = parser.parse_args()

    check_hitemp_credentials(args)
    groups = parse_groups(args.groups)

    for case in selected_cases(args.case):
        print(f"RADIS {args.databank.upper()} {case.name}: {case.note}", flush=True)
        spectrum, elapsed_s = run_radis_case(case, args)
        spectral_csv, summary_csv, plot_png = write_outputs(case, args.databank, elapsed_s, groups, spectrum, args.out_dir)
        print(f"  wrote {spectral_csv}", flush=True)
        print(f"  wrote {summary_csv}", flush=True)
        print(f"  wrote {plot_png}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
