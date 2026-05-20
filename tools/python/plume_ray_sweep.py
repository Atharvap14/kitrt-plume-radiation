#!/usr/bin/env python3
"""Benchmark plume-radiation ray sampling on MLX GPU, CPU-MPI, or CPU.

The estimator is Monte Carlo over upper-hemisphere ray directions:

    q(r) = 2*pi * E[I(r, omega) * cos(theta)]

It reports the standard error of that flux estimate for each base-radius
sample. The synthetic field is intentionally analytic so all backends run the
same workload without file I/O.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


SIGMA_SB = 5.670374419e-8


@dataclass(frozen=True)
class SweepConfig:
    rays: Sequence[int]
    base_samples: int
    base_radius: float
    max_distance: float
    step: float
    chunk_size: int
    seed: int
    target_rel_se: float
    z_max: float
    r_max: float
    ambient_temperature: float
    peak_temperature: float
    z_center: float
    z_scale: float
    r_scale: float
    kappa_floor: float
    kappa_peak: float
    kappa_z_scale: float
    kappa_r_scale: float
    output_csv: Optional[str]


def parse_rays(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 1:
            raise ValueError("Ray counts must be greater than 1")
        values.append(value)
    if not values:
        raise ValueError("At least one ray count is required")
    return values


def base_radii(config: SweepConfig) -> np.ndarray:
    if config.base_samples == 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(0.0, config.base_radius, config.base_samples, dtype=np.float32)


def ray_samples(num_rays: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + num_rays * 1009)
    mu = rng.random(num_rays, dtype=np.float32)
    phi = rng.random(num_rays, dtype=np.float32) * np.float32(2.0 * math.pi)
    return mu, phi


def _finalize_stats(sum_y: np.ndarray, sum_y2: np.ndarray, num_rays: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.float64(2.0 * math.pi) * sum_y / np.float64(num_rays)
    variance_y = (sum_y2 - sum_y * sum_y / np.float64(num_rays)) / np.float64(max(num_rays - 1, 1))
    variance_y = np.maximum(variance_y, 0.0)
    standard_error = np.float64(2.0 * math.pi) * np.sqrt(variance_y / np.float64(num_rays))
    relative_standard_error = standard_error / np.maximum(np.abs(flux), 1e-300)
    return flux, standard_error, relative_standard_error


def _record(
    backend: str,
    rays: int,
    elapsed_s: float,
    flux: np.ndarray,
    standard_error: np.ndarray,
    relative_standard_error: np.ndarray,
    ranks: int = 1,
    target_rel_se: float = 1.0e-3,
) -> dict:
    return {
        "backend": backend,
        "ranks": ranks,
        "rays": rays,
        "elapsed_s": elapsed_s,
        "rays_per_s": rays / elapsed_s if elapsed_s > 0.0 else float("inf"),
        "mean_flux": float(np.mean(flux)),
        "min_flux": float(np.min(flux)),
        "max_flux": float(np.max(flux)),
        "max_standard_error": float(np.max(standard_error)),
        "mean_relative_standard_error": float(np.mean(relative_standard_error)),
        "max_relative_standard_error": float(np.max(relative_standard_error)),
        "target_met": bool(np.max(relative_standard_error) <= target_rel_se),
    }


def _print_records(records: Iterable[dict], rank: int = 0) -> None:
    if rank != 0:
        return
    print(
        "backend,ranks,rays,elapsed_s,rays_per_s,mean_flux,max_abs_se,"
        "mean_rel_se,max_rel_se,target_met",
        flush=True,
    )
    for row in records:
        print(
            f"{row['backend']},{row['ranks']},{row['rays']},"
            f"{row['elapsed_s']:.6f},{row['rays_per_s']:.2f},"
            f"{row['mean_flux']:.9e},{row['max_standard_error']:.9e},"
            f"{row['mean_relative_standard_error']:.9e},"
            f"{row['max_relative_standard_error']:.9e},{row['target_met']}",
            flush=True,
        )


def _write_csv(path: str, records: Sequence[dict]) -> None:
    fieldnames = [
        "backend",
        "ranks",
        "rays",
        "elapsed_s",
        "rays_per_s",
        "mean_flux",
        "min_flux",
        "max_flux",
        "max_standard_error",
        "mean_relative_standard_error",
        "max_relative_standard_error",
        "target_met",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _integrate_numpy(mu: np.ndarray, phi: np.ndarray, config: SweepConfig, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    radii64 = radii.astype(np.float64)
    sum_y = np.zeros(radii64.shape[0], dtype=np.float64)
    sum_y2 = np.zeros(radii64.shape[0], dtype=np.float64)
    num_steps = int(math.ceil(config.max_distance / config.step))

    for start in range(0, mu.shape[0], config.chunk_size):
        stop = min(start + config.chunk_size, mu.shape[0])
        mu_c = mu[start:stop].astype(np.float64)
        phi_c = phi[start:stop].astype(np.float64)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu_c * mu_c))
        cos_phi = np.cos(phi_c)
        sin_phi = np.sin(phi_c)

        intensity = np.zeros((radii64.shape[0], stop - start), dtype=np.float64)
        transmittance = np.ones_like(intensity)

        for idx_step in range(num_steps):
            s = idx_step * config.step
            ds = min(config.step, config.max_distance - s)
            if ds <= 0.0:
                break
            smid = s + 0.5 * ds
            z = smid * mu_c[None, :]
            x = radii64[:, None] + smid * sin_theta[None, :] * cos_phi[None, :]
            y = smid * sin_theta[None, :] * sin_phi[None, :]
            r = np.sqrt(x * x + y * y)
            inside = (z <= config.z_max) & (r <= config.r_max)

            temperature = config.ambient_temperature + config.peak_temperature * np.exp(
                -((z - config.z_center) / config.z_scale) ** 2 - (r / config.r_scale) ** 2
            )
            kappa = config.kappa_floor + config.kappa_peak * np.exp(
                -z / config.kappa_z_scale - (r / config.kappa_r_scale) ** 2
            )
            temperature = np.where(inside, temperature, 0.0)
            kappa = np.where(inside, kappa, 0.0)

            attenuation = np.exp(-np.minimum(kappa * ds, 700.0))
            blackbody = (SIGMA_SB / math.pi) * temperature**4
            intensity += transmittance * blackbody * (1.0 - attenuation)
            transmittance *= attenuation

        contribution = intensity * mu_c[None, :]
        sum_y += np.sum(contribution, axis=1)
        sum_y2 += np.sum(contribution * contribution, axis=1)

    return sum_y, sum_y2


def run_cpu(config: SweepConfig, backend_name: str = "cpu") -> List[dict]:
    radii = base_radii(config)
    records: List[dict] = []
    for rays in config.rays:
        mu, phi = ray_samples(rays, config.seed)
        t0 = time.perf_counter()
        sum_y, sum_y2 = _integrate_numpy(mu, phi, config, radii)
        elapsed = time.perf_counter() - t0
        flux, se, rel_se = _finalize_stats(sum_y, sum_y2, rays)
        records.append(_record(backend_name, rays, elapsed, flux, se, rel_se, target_rel_se=config.target_rel_se))
    return records


def run_mpi(config: SweepConfig) -> List[dict]:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    radii = base_radii(config)
    records: List[dict] = []

    for rays in config.rays:
        mu, phi = ray_samples(rays, config.seed)
        start = rank * rays // size
        stop = (rank + 1) * rays // size
        comm.Barrier()
        t0 = time.perf_counter()
        local_sum_y, local_sum_y2 = _integrate_numpy(mu[start:stop], phi[start:stop], config, radii)
        elapsed_local = time.perf_counter() - t0
        sum_y = np.zeros_like(local_sum_y)
        sum_y2 = np.zeros_like(local_sum_y2)
        comm.Allreduce(local_sum_y, sum_y, op=MPI.SUM)
        comm.Allreduce(local_sum_y2, sum_y2, op=MPI.SUM)
        elapsed = comm.allreduce(elapsed_local, op=MPI.MAX)

        if rank == 0:
            flux, se, rel_se = _finalize_stats(sum_y, sum_y2, rays)
            records.append(_record("mpi", rays, elapsed, flux, se, rel_se, ranks=size, target_rel_se=config.target_rel_se))
    return records


def run_mlx(config: SweepConfig) -> List[dict]:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise SystemExit("MLX is not installed. Run: python3 -m pip install --user mlx") from exc

    mx.set_default_device(mx.gpu)
    radii = base_radii(config)
    radii_mx = mx.array(radii, dtype=mx.float32)
    records: List[dict] = []
    num_steps = int(math.ceil(config.max_distance / config.step))

    for rays in config.rays:
        mu_np, phi_np = ray_samples(rays, config.seed)
        sum_y = mx.zeros((radii.shape[0],), dtype=mx.float32)
        sum_y2 = mx.zeros((radii.shape[0],), dtype=mx.float32)
        mx.eval(sum_y, sum_y2)

        t0 = time.perf_counter()
        for start in range(0, rays, config.chunk_size):
            stop = min(start + config.chunk_size, rays)
            mu = mx.array(mu_np[start:stop], dtype=mx.float32)
            phi = mx.array(phi_np[start:stop], dtype=mx.float32)
            sin_theta = mx.sqrt(mx.maximum(mx.array(0.0, dtype=mx.float32), 1.0 - mu * mu))
            cos_phi = mx.cos(phi)
            sin_phi = mx.sin(phi)
            intensity = mx.zeros((radii.shape[0], stop - start), dtype=mx.float32)
            transmittance = mx.ones_like(intensity)

            for idx_step in range(num_steps):
                s = idx_step * config.step
                ds = min(config.step, config.max_distance - s)
                if ds <= 0.0:
                    break
                smid = np.float32(s + 0.5 * ds)
                z = smid * mu[None, :]
                x = radii_mx[:, None] + smid * sin_theta[None, :] * cos_phi[None, :]
                y = smid * sin_theta[None, :] * sin_phi[None, :]
                r = mx.sqrt(x * x + y * y)
                inside = (z <= config.z_max) & (r <= config.r_max)

                temperature = config.ambient_temperature + config.peak_temperature * mx.exp(
                    -((z - config.z_center) / config.z_scale) ** 2 - (r / config.r_scale) ** 2
                )
                kappa = config.kappa_floor + config.kappa_peak * mx.exp(
                    -z / config.kappa_z_scale - (r / config.kappa_r_scale) ** 2
                )
                temperature = mx.where(inside, temperature, 0.0)
                kappa = mx.where(inside, kappa, 0.0)

                attenuation = mx.exp(-mx.minimum(kappa * ds, 700.0))
                blackbody = np.float32(SIGMA_SB / math.pi) * temperature**4
                intensity = intensity + transmittance * blackbody * (1.0 - attenuation)
                transmittance = transmittance * attenuation
                if (idx_step + 1) % 32 == 0:
                    mx.eval(intensity, transmittance)

            contribution = intensity * mu[None, :]
            sum_y = sum_y + mx.sum(contribution, axis=1)
            sum_y2 = sum_y2 + mx.sum(contribution * contribution, axis=1)
            mx.eval(sum_y, sum_y2)

        mx.eval(sum_y, sum_y2)
        elapsed = time.perf_counter() - t0
        sum_y_np = np.array(sum_y, dtype=np.float64)
        sum_y2_np = np.array(sum_y2, dtype=np.float64)
        flux, se, rel_se = _finalize_stats(sum_y_np, sum_y2_np, rays)
        records.append(_record("mlx", rays, elapsed, flux, se, rel_se, target_rel_se=config.target_rel_se))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "mpi", "cpu"], required=True)
    parser.add_argument("--rays", default="1024,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--base-samples", type=int, default=8)
    parser.add_argument("--base-radius", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.03)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--target-rel-se", type=float, default=1.0e-3)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument("--ambient-temperature", type=float, default=450.0)
    parser.add_argument("--peak-temperature", type=float, default=1900.0)
    parser.add_argument("--z-center", type=float, default=0.7)
    parser.add_argument("--z-scale", type=float, default=0.9)
    parser.add_argument("--r-scale", type=float, default=0.55)
    parser.add_argument("--kappa-floor", type=float, default=0.01)
    parser.add_argument("--kappa-peak", type=float, default=0.35)
    parser.add_argument("--kappa-z-scale", type=float, default=1.2)
    parser.add_argument("--kappa-r-scale", type=float, default=0.8)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> SweepConfig:
    if args.base_samples <= 0:
        raise SystemExit("--base-samples must be positive")
    if args.step <= 0.0:
        raise SystemExit("--step must be positive")
    if args.max_distance <= 0.0:
        raise SystemExit("--max-distance must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    return SweepConfig(
        rays=parse_rays(args.rays),
        base_samples=args.base_samples,
        base_radius=args.base_radius,
        max_distance=args.max_distance,
        step=args.step,
        chunk_size=args.chunk_size,
        seed=args.seed,
        target_rel_se=args.target_rel_se,
        z_max=args.z_max,
        r_max=args.r_max,
        ambient_temperature=args.ambient_temperature,
        peak_temperature=args.peak_temperature,
        z_center=args.z_center,
        z_scale=args.z_scale,
        r_scale=args.r_scale,
        kappa_floor=args.kappa_floor,
        kappa_peak=args.kappa_peak,
        kappa_z_scale=args.kappa_z_scale,
        kappa_r_scale=args.kappa_r_scale,
        output_csv=args.output_csv,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    rank = 0
    if args.backend == "mlx":
        records = run_mlx(config)
    elif args.backend == "mpi":
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        records = run_mpi(config)
    else:
        records = run_cpu(config)

    _print_records(records, rank=rank)
    if rank == 0 and config.output_csv:
        _write_csv(config.output_csv, records)


if __name__ == "__main__":
    main()
