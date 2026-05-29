#!/usr/bin/env python3
"""
SOC-style diagnostics: branching ratio (excited-area ratio) and PSD of spatially averaged signal.
Runs a fixed-duration simulation with no visualization (no pygame window).
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wave_sim import Params, run_soc_simulation


def periodogram_psd(x: np.ndarray, sample_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-segment Hanning-windowed periodogram; one-sided frequencies."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    if n < 4:
        return np.array([]), np.array([])
    w = np.hanning(n)
    xw = x * w
    scale = np.sum(w**2)
    freqs = np.fft.rfftfreq(n, d=sample_dt)
    spec = np.abs(np.fft.rfft(xw)) ** 2 / (sample_dt * n * scale)
    spec[0] = 0.0
    if n > 1 and spec.size > 1:
        spec[1:-1] *= 2.0
    return freqs, spec


def main() -> None:
    parser = argparse.ArgumentParser(description="SOC analysis: branching ratio + PSD (no display)")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation time (s)")
    parser.add_argument("--sample-dt", type=float, default=0.02, help="Time between samples (s)")
    parser.add_argument("--branch-lag", type=float, default=0.1, help="Lag for area ratio (s)")
    parser.add_argument(
        "--q-threshold-frac",
        type=float,
        default=0.2,
        help="Excited if Q_e > qmax_e * this (default 1/5)",
    )
    parser.add_argument(
        "--psd-signal",
        choices=("phi", "Q"),
        default="phi",
        help="Spatial mean of phi_e or mean Q_e for PSD",
    )
    parser.add_argument("--out-dir", type=str, default="soc_out", help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = Params()
    data = run_soc_simulation(
        p,
        duration_s=args.duration,
        sample_dt=args.sample_dt,
        branch_lag_s=args.branch_lag,
        q_threshold_frac=args.q_threshold_frac,
    )

    t = data["t"]
    area = data["area_excited"]
    br = data["branch_ratio"]
    sample_dt = float(data["sample_dt"])

    fig1, ax1 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax1.plot(t, br, lw=0.8, color="C0")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Area$(t)$ / Area$(t-\Delta t)$")
    ax1.set_title(
        f"Branching proxy (excited: $Q_e > {data['q_threshold']:.3g}$), "
        f"$\\Delta t$={args.branch_lag:.3g} s, sample_dt={sample_dt:.3g} s"
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5)
    p1 = os.path.join(args.out_dir, "branching_ratio.png")
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    sig = data["mean_phi_e"] if args.psd_signal == "phi" else data["mean_Q_e"]
    freqs, psd = periodogram_psd(sig, sample_dt)
    fig2, ax2 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2.loglog(freqs[1:], np.maximum(psd[1:], 1e-30), lw=0.8, color="C1")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (one-sided)")
    ax2.set_title(f"PSD of spatial mean ({args.psd_signal}), N={len(sig)}, Hanning window")
    ax2.grid(True, which="both", alpha=0.3)
    p2 = os.path.join(args.out_dir, "psd_loglog.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    np.savez_compressed(
        os.path.join(args.out_dir, "soc_timeseries.npz"),
        t=t,
        area_excited=area,
        branch_ratio=br,
        mean_phi_e=data["mean_phi_e"],
        mean_Q_e=data["mean_Q_e"],
        sample_dt=sample_dt,
        branch_lag_s=args.branch_lag,
        q_threshold=data["q_threshold"],
    )
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {os.path.join(args.out_dir, 'soc_timeseries.npz')}")


if __name__ == "__main__":
    main()
