import argparse
import os
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Params:
    # --- Grid / numerics ---
    n: int = 220  # grid resolution (n x n)
    dx: float = 1.0
    fps: int = 60
    steps_per_frame: int = 2  # increase for better stability at higher gamma*r (faster waves)

    # --- Neural field (wave equation) parameters (edit these a lot) ---
    # Matches the paper operator:
    #   [ 1/gamma^2 d2/dt2 + 2/gamma d/dt + 1 - r^2 Lap ] phi = Q
    # implemented as:
    #   phi_tt = gamma^2 * r^2 Lap(phi) - 2*gamma*phi_t - gamma^2*phi + gamma^2*Q
    #
    # Wave-like tuning (heuristic):
    #   - Laplacian strength ~ (gamma*r)^2 on phi: increase r_e for faster / clearer fronts.
    #   - Temporal damping ~ 2*gamma on phi_t: decrease gamma_e for less overdamping (more ringing).
    #   gamma and r trade off: do not only lower gamma (that also shrinks gamma^2*r^2); pair with larger r_e.
    # Criticality (branching ~1): nudge excitation toward marginal stability (nu_ee up slightly, |nu_ei| down slightly)
    # and avoid saturating Q with huge drive or noise.
    gamma_e: float = 8.5  # s^-1 (lower than paper default → weaker damping term 2*gamma*phi_t)
    r_e: float = 1.05  # grid units; larger → more wave-like propagation vs local decay
    gamma_i: float = 8.5  # s^-1
    r_i: float = 0.0  # paper often treats i as local (no propagation); keep 0 for local inhibitory field

    # --- Synaptodendritic filter parameters (edit these a lot) ---
    # Paper Eq (3):
    #   [ 1/(alpha*beta) d2/dt2 + (1/alpha + 1/beta) d/dt + 1 ] V = P
    # equivalently:
    #   V_tt + (alpha+beta)V_t + alpha*beta*V = alpha*beta*P
    alpha: float = 83.33  # s^-1
    beta_over_alpha: float = 9.23  # so beta = alpha * beta_over_alpha

    # --- Sigmoid parameters for firing rates Q (edit these a lot) ---
    qmax_e: float = 340.0  # s^-1
    theta_e: float = 0.01292  # V
    sigma_e: float = 0.0034  # V
    qmax_i: float = 340.0  # s^-1
    theta_i: float = 0.01292  # V
    sigma_i: float = 0.0034  # V

    # --- Coupling / gains (edit these a lot) ---
    # Simplified local coupling (paper Eq 4, dropping thalamic populations for now):
    #   P_e = nu_ee * phi_e + nu_ei * phi_i + drive
    #   P_i = nu_ie * phi_e + nu_ii * phi_i + drive_i
    nu_ee: float = 3.25e-3  # V*s (slightly ↑ toward critical excitation)
    nu_ei: float = -5.85e-3  # V*s (slightly ↓ |inhibition| for E/I nearer marginal)
    nu_ie: float = 3.03e-3  # V*s  (placeholder; tune)
    nu_ii: float = -6.00e-3  # V*s (placeholder; tune)
    # Note: drive terms enter P (units of Volts). Keep them near theta (~1e-2 V), not O(1).
    drive_e: float = 0.0095  # V (slightly below prior to reduce saturation; tune with nu_* for criticality)
    drive_i: float = 0.0

    # --- Transient perturbation / start-up "kick" ---
    # Without a perturbation, the (spatially uniform) fixed point can sit there indefinitely.
    # This kick is transient and designed to seed spatial structure.
    kick_enabled_on_start: bool = True
    kick_mode: str = "gaussian_phi_e"  # "gaussian_phi_e" | "drive_pulse" | "drive_noise"
    kick_duration_s: float = 0.25
    kick_amp: float = 40.0  # amplitude of the kick (meaning depends on mode)
    kick_sigma_cells: float = 6.0  # spatial width for gaussian kick (in grid cells)
    kick_center_x_frac: float = 0.5
    kick_center_y_frac: float = 0.65

    # --- Stochastic drive (SOC-friendly) ---
    # Slow, weak colored noise + sparse local "shots" to seed avalanches.
    noise_enabled: bool = True
    noise_seed: int = 1
    # OU noise (global, colored)
    ou_enabled: bool = True
    ou_tau_s: float = 2.0
    ou_sigma_v: float = 2.0e-4  # volts
    ou_mu_v: float = 0.0
    # Poisson shot noise (spatially localized)
    shots_enabled: bool = True
    shots_rate_hz: float = 3.5  # events / second — lower so fronts from kicks/propagation dominate
    shots_amp_v: float = 7.0e-3  # volts (moderate; large shots mask wave geometry)
    shots_sigma_cells: float = 3.2  # spatial spread of each event
    shots_decay_tau_s: float = 0.30  # exponential decay of accumulated shot field
    events_hud_ema_tau_s: float = 1.0  # smoothing for displayed events/sec

    # --- Visualization ---
    window_px: int = 860
    wave_gain: float = 1.8  # scales visible contrast for phi_e
    bg_gray: int = 35  # background gray (0-255)
    render_deviation: bool = True  # visualize phi_e - mean(phi_e) to avoid DC washout
    # 5-point Laplacian makes wavefronts look square/diamond (grid anisotropy). Use 9-point for rounder rings.
    laplacian_9pt: bool = True


def laplacian_neumann(phi: np.ndarray) -> np.ndarray:
    """
    5-point Laplacian with reflective (Neumann) boundaries:
    normal derivative = 0, implemented by edge replication.
    """
    # pad with edge values to enforce zero normal derivative
    p = np.pad(phi, 1, mode="edge")
    return (
        p[2:, 1:-1]
        + p[:-2, 1:-1]
        + p[1:-1, 2:]
        + p[1:-1, :-2]
        - 4.0 * p[1:-1, 1:-1]
    )


def laplacian_neumann_9pt(phi: np.ndarray) -> np.ndarray:
    """
    9-point (isotropic) Laplacian on a square grid with Neumann boundaries (edge pad).
    Reduces the "boxy / diamond" rings around point sources vs the 5-point stencil.
    Same order of accuracy O(h^2); different anisotropic error on the grid.
    """
    p = np.pad(phi, 1, mode="edge")
    c = p[1:-1, 1:-1]
    ax = p[0:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, 0:-2] + p[1:-1, 2:]
    dg = p[0:-2, 0:-2] + p[0:-2, 2:] + p[2:, 0:-2] + p[2:, 2:]
    return (4.0 * ax + dg - 20.0 * c) / 6.0


def laplacian_neumann_dispatch(phi: np.ndarray, use_9pt: bool) -> np.ndarray:
    return laplacian_neumann_9pt(phi) if use_9pt else laplacian_neumann(phi)


def sigmoid(v: np.ndarray, qmax: float, theta: float, sigma: float) -> np.ndarray:
    # numerically stable-ish sigmoid; sigma is positive std dev
    s = max(1e-9, float(sigma))
    x = (v - float(theta)) / s
    x = np.clip(x, -60.0, 60.0)
    return (float(qmax) / (1.0 + np.exp(-x))).astype(np.float32)


def render_field_white_on_gray(phi: np.ndarray, gain: float, bg_gray: int, deviation: bool) -> np.ndarray:
    """
    Dark grey background with white wave intensity.
    Uses |phi| so both polarities are visible as "white waves".
    """
    bg = int(np.clip(bg_gray, 0, 255))
    x0 = phi - float(np.mean(phi)) if deviation else phi
    # auto-scale by spatial std so changes remain visible across parameter sweeps
    s = float(np.std(x0))
    s = max(1e-6, s)
    x = np.tanh(float(gain) * (np.abs(x0) / s)).astype(np.float32)  # [0, 1)
    intensity = (255.0 * x).astype(np.uint8)
    base = np.full(phi.shape, bg, dtype=np.uint8)
    # blend to white (do arithmetic in wider dtype to avoid uint8 overflow)
    scale = np.uint16(255 - bg)
    out = base.astype(np.uint16) + (scale * intensity.astype(np.uint16) // np.uint16(255))
    out8 = out.astype(np.uint8)
    return np.dstack([out8, out8, out8])


def run_soc_simulation(
    p: Params,
    duration_s: float,
    sample_dt: float,
    branch_lag_s: float,
    q_threshold_frac: float = 0.2,
) -> dict[str, Any]:
    """
    Run the neural-field dynamics without pygame; record time series for SOC-style analysis.

    Excited area: grid cells where Q_e > qmax_e * q_threshold_frac (default 1/5).
    Branching proxy: area(t) / area(t - branch_lag_s), undefined when past area is ~0.
    """
    n = p.n
    phi_e = np.zeros((n, n), dtype=np.float32)
    dphi_e = np.zeros((n, n), dtype=np.float32)
    phi_i = np.zeros((n, n), dtype=np.float32)
    dphi_i = np.zeros((n, n), dtype=np.float32)
    V_e = np.zeros((n, n), dtype=np.float32)
    dV_e = np.zeros((n, n), dtype=np.float32)
    V_i = np.zeros((n, n), dtype=np.float32)
    dV_i = np.zeros((n, n), dtype=np.float32)

    rng = np.random.default_rng(int(p.noise_seed))
    xs = np.arange(n, dtype=np.float32)
    ys = np.arange(n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    cx = float(p.kick_center_x_frac) * float(n - 1)
    cy = float(p.kick_center_y_frac) * float(n - 1)
    sig = max(1e-3, float(p.kick_sigma_cells))
    kick_gauss = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig)).astype(np.float32)

    kick_time_remaining = p.kick_duration_s if p.kick_enabled_on_start else 0.0
    if p.kick_enabled_on_start and p.kick_mode == "gaussian_phi_e":
        phi_e[:] += (float(p.kick_amp) * kick_gauss).astype(np.float32)

    beta0 = p.alpha * p.beta_over_alpha
    max_rate = max(p.alpha + beta0, p.gamma_e, p.gamma_i, 1.0)
    dt_target = 0.05 / float(max_rate)
    dt_frame = 1.0 / p.fps
    steps = max(int(p.steps_per_frame), int(math.ceil(dt_frame / dt_target)))
    dt = dt_frame / float(steps)

    ou_x = float(p.ou_mu_v)
    shot_field = np.zeros((n, n), dtype=np.float32)

    def add_shot_event() -> None:
        sig_s = max(0.5, float(p.shots_sigma_cells))
        rad = int(math.ceil(4.0 * sig_s))
        x0 = int(rng.integers(0, n))
        y0 = int(rng.integers(0, n))
        x1 = max(0, x0 - rad)
        x2 = min(n, x0 + rad + 1)
        y1 = max(0, y0 - rad)
        y2 = min(n, y0 + rad + 1)
        xs_ = np.arange(x1, x2, dtype=np.float32) - float(x0)
        ys_ = np.arange(y1, y2, dtype=np.float32) - float(y0)
        Xs, Ys = np.meshgrid(xs_, ys_, indexing="xy")
        kern = np.exp(-(Xs * Xs + Ys * Ys) / (2.0 * sig_s * sig_s)).astype(np.float32)
        shot_field[y1:y2, x1:x2] += (float(p.shots_amp_v) * kern).astype(np.float32)

    thr_Q = float(p.qmax_e * q_threshold_frac)
    cell_area = float(p.dx * p.dx)

    t = 0.0
    next_sample_t = 0.0
    t_samples: list[float] = []
    area_samples: list[float] = []
    mean_phi_samples: list[float] = []
    mean_Q_samples: list[float] = []

    def record_sample(sample_time: float) -> None:
        Q_e = sigmoid(V_e, p.qmax_e, p.theta_e, p.sigma_e)
        excited = Q_e > thr_Q
        area = float(np.count_nonzero(excited)) * cell_area
        t_samples.append(sample_time)
        area_samples.append(area)
        mean_phi_samples.append(float(np.mean(phi_e)))
        mean_Q_samples.append(float(np.mean(Q_e)))

    while next_sample_t <= duration_s + 1e-12 and t + 1e-15 >= next_sample_t:
        record_sample(next_sample_t)
        next_sample_t += sample_dt

    while t < duration_s - 1e-15:
        for _ in range(steps):
            beta = p.alpha * p.beta_over_alpha
            drive_e = float(p.drive_e)
            if kick_time_remaining > 0.0:
                if p.kick_mode == "drive_pulse":
                    drive_e += float(p.kick_amp)
                elif p.kick_mode == "drive_noise":
                    drive_e += float(p.kick_amp) * float(rng.standard_normal())

            if p.noise_enabled:
                if p.ou_enabled:
                    tau = max(1e-6, float(p.ou_tau_s))
                    ou_x += ((float(p.ou_mu_v) - ou_x) / tau) * dt + float(p.ou_sigma_v) * math.sqrt(dt) * float(
                        rng.standard_normal()
                    )
                if p.shots_enabled:
                    tau_s = max(1e-6, float(p.shots_decay_tau_s))
                    shot_field *= float(math.exp(-dt / tau_s))
                    if float(p.shots_rate_hz) > 0.0:
                        k = int(rng.poisson(float(p.shots_rate_hz) * dt))
                        for _ev in range(k):
                            add_shot_event()

            drive_field = drive_e + (ou_x if (p.noise_enabled and p.ou_enabled) else 0.0)
            if p.noise_enabled and p.shots_enabled:
                P_e = (p.nu_ee * phi_e + p.nu_ei * phi_i + drive_field + shot_field).astype(np.float32)
            else:
                P_e = (p.nu_ee * phi_e + p.nu_ei * phi_i + drive_field).astype(np.float32)
            P_i = (p.nu_ie * phi_e + p.nu_ii * phi_i + p.drive_i).astype(np.float32)

            ddV_e = (p.alpha * beta) * (P_e - V_e) - (p.alpha + beta) * dV_e
            ddV_i = (p.alpha * beta) * (P_i - V_i) - (p.alpha + beta) * dV_i
            dV_e += dt * ddV_e
            V_e += dt * dV_e
            dV_i += dt * ddV_i
            V_i += dt * dV_i

            Q_e = sigmoid(V_e, p.qmax_e, p.theta_e, p.sigma_e)
            Q_i = sigmoid(V_i, p.qmax_i, p.theta_i, p.sigma_i)

            lap_e = laplacian_neumann_dispatch(phi_e, p.laplacian_9pt) / (p.dx * p.dx)
            lap_i = (
                laplacian_neumann_dispatch(phi_i, p.laplacian_9pt) / (p.dx * p.dx) if p.r_i != 0.0 else 0.0
            )

            ddphi_e = (
                (p.gamma_e * p.gamma_e) * (p.r_e * p.r_e) * lap_e
                - 2.0 * p.gamma_e * dphi_e
                - (p.gamma_e * p.gamma_e) * phi_e
                + (p.gamma_e * p.gamma_e) * Q_e
            )
            ddphi_i = (
                (p.gamma_i * p.gamma_i) * (p.r_i * p.r_i) * lap_i
                - 2.0 * p.gamma_i * dphi_i
                - (p.gamma_i * p.gamma_i) * phi_i
                + (p.gamma_i * p.gamma_i) * Q_i
            )

            dphi_e += dt * ddphi_e
            phi_e += dt * dphi_e
            dphi_i += dt * ddphi_i
            phi_i += dt * dphi_i
            t += dt
            if kick_time_remaining > 0.0:
                kick_time_remaining = max(0.0, kick_time_remaining - dt)

            while next_sample_t <= duration_s + 1e-12 and t + 1e-12 >= next_sample_t:
                record_sample(next_sample_t)
                next_sample_t += sample_dt

    t_arr = np.asarray(t_samples, dtype=np.float64)
    area = np.asarray(area_samples, dtype=np.float64)
    mean_phi = np.asarray(mean_phi_samples, dtype=np.float64)
    mean_Q = np.asarray(mean_Q_samples, dtype=np.float64)

    lag_n = max(1, int(round(branch_lag_s / sample_dt)))
    branch_ratio = np.full_like(area, np.nan, dtype=np.float64)
    eps = 1e-18
    for i in range(lag_n, len(area)):
        ap = area[i - lag_n]
        branch_ratio[i] = area[i] / ap if ap > eps else np.nan

    return {
        "t": t_arr,
        "area_excited": area,
        "mean_phi_e": mean_phi,
        "mean_Q_e": mean_Q,
        "branch_ratio": branch_ratio,
        "branch_lag_s": branch_lag_s,
        "lag_n_samples": lag_n,
        "sample_dt": sample_dt,
        "q_threshold": thr_Q,
        "integrator_dt": dt,
    }


def main() -> None:
    import pygame

    parser = argparse.ArgumentParser(description="2D damped wave equation simulator (reflective boundaries)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a visible window (uses SDL dummy video driver).",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default="",
        help="When set, save rendered PNG frames to this directory (works in headless too).",
    )
    parser.add_argument(
        "--save-frames-every",
        type=int,
        default=10,
        help="Save one frame every N display frames (when --save-frames is set).",
    )
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=5.0,
        help="When --headless, run for this many seconds of simulated time.",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default="",
        help="When --headless, save phi/v snapshots to this .npz path at the end.",
    )
    args = parser.parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    p = Params()

    pygame.init()
    pygame.display.set_caption("2D damped wave equation (reflective box)")
    try:
        screen = pygame.display.set_mode((p.window_px, p.window_px))
    except pygame.error as e:
        raise SystemExit(
            "Could not open a window. If you're running in a headless environment, "
            "re-run with `python wave_sim.py --headless`."
        ) from e
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 14) or pygame.font.SysFont(None, 14)

    n = p.n

    # State for excitatory field
    phi_e = np.zeros((n, n), dtype=np.float32)
    dphi_e = np.zeros((n, n), dtype=np.float32)  # phi_t

    # State for inhibitory field
    phi_i = np.zeros((n, n), dtype=np.float32)
    dphi_i = np.zeros((n, n), dtype=np.float32)

    # Local membrane potentials (synaptodendritic filtered)
    V_e = np.zeros((n, n), dtype=np.float32)
    dV_e = np.zeros((n, n), dtype=np.float32)
    V_i = np.zeros((n, n), dtype=np.float32)
    dV_i = np.zeros((n, n), dtype=np.float32)

    paused = False
    t = 0.0

    rng = np.random.default_rng(int(p.noise_seed))

    # Precompute a spatial gaussian bump for kick modes that need it.
    xs = np.arange(n, dtype=np.float32)
    ys = np.arange(n, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    cx = float(p.kick_center_x_frac) * float(n - 1)
    cy = float(p.kick_center_y_frac) * float(n - 1)
    sig = max(1e-3, float(p.kick_sigma_cells))
    kick_gauss = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sig * sig)).astype(np.float32)

    def reset() -> None:
        nonlocal t
        phi_e.fill(0.0)
        dphi_e.fill(0.0)
        phi_i.fill(0.0)
        dphi_i.fill(0.0)
        V_e.fill(0.0)
        dV_e.fill(0.0)
        V_i.fill(0.0)
        dV_i.fill(0.0)
        t = 0.0

    kick_time_remaining = p.kick_duration_s if p.kick_enabled_on_start else 0.0

    def trigger_kick() -> None:
        nonlocal kick_time_remaining
        kick_time_remaining = max(0.0, float(p.kick_duration_s))
        # For an instantaneous kick, directly seed the field once.
        if p.kick_mode == "gaussian_phi_e":
            phi_e[:] += (float(p.kick_amp) * kick_gauss).astype(np.float32)

    # CFL-ish stability: dt <= dx / (c*sqrt(2)) for undamped wave
    # We choose dt from fps and steps_per_frame, and rely on user to keep parameters reasonable.
    dt_frame = 1.0 / p.fps
    # The synaptodendritic ODE is stiff when alpha/beta are large; use automatic substepping.
    # Heuristic: keep dt_sub <= 0.05 / max_rate to avoid explicit-Euler blowups.
    beta0 = p.alpha * p.beta_over_alpha
    max_rate = max(p.alpha + beta0, p.gamma_e, p.gamma_i, 1.0)
    dt_target = 0.05 / float(max_rate)
    steps = max(int(p.steps_per_frame), int(math.ceil(dt_frame / dt_target)))
    dt = dt_frame / float(steps)

    running = True
    headless_steps_remaining = None
    if args.headless:
        headless_steps_remaining = int(max(1, round(args.headless_seconds / dt)))

    if p.kick_enabled_on_start:
        trigger_kick()

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
    frame_idx = 0

    # Noise state
    ou_x = float(p.ou_mu_v)  # global OU state (Volts)
    shot_field = np.zeros((n, n), dtype=np.float32)  # spatial field added to drive_e (Volts)
    events_in_frame = 0
    events_per_s_ema = 0.0

    def add_shot_event() -> None:
        sig_s = max(0.5, float(p.shots_sigma_cells))
        # 4σ avoids a hard rectangular cutoff at the patch edge (tiny artifact vs 3σ).
        rad = int(math.ceil(4.0 * sig_s))
        x0 = int(rng.integers(0, n))
        y0 = int(rng.integers(0, n))
        x1 = max(0, x0 - rad)
        x2 = min(n, x0 + rad + 1)
        y1 = max(0, y0 - rad)
        y2 = min(n, y0 + rad + 1)
        xs = np.arange(x1, x2, dtype=np.float32) - float(x0)
        ys = np.arange(y1, y2, dtype=np.float32) - float(y0)
        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        kern = np.exp(-(Xs * Xs + Ys * Ys) / (2.0 * sig_s * sig_s)).astype(np.float32)
        shot_field[y1:y2, x1:x2] += (float(p.shots_amp_v) * kern).astype(np.float32)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key == pygame.K_k:
                    trigger_kick()
                elif event.key == pygame.K_n:
                    p.noise_enabled = not p.noise_enabled
                elif event.key == pygame.K_LEFTBRACKET:
                    p.gamma_e = max(0.0, p.gamma_e * 0.85)
                elif event.key == pygame.K_RIGHTBRACKET:
                    p.gamma_e = p.gamma_e / 0.85
                elif event.key == pygame.K_MINUS:
                    p.qmax_e *= 0.85
                elif event.key == pygame.K_EQUALS:
                    p.qmax_e /= 0.85
                elif event.key == pygame.K_COMMA:
                    p.nu_ee *= 0.85
                elif event.key == pygame.K_PERIOD:
                    p.nu_ee /= 0.85

        if not paused:
            events_in_frame = 0
            for _ in range(steps):
                beta = p.alpha * p.beta_over_alpha

                # Presynaptic inputs (simplified local coupling, paper Eq 4 reduced to e/i only)
                drive_e = float(p.drive_e)
                if kick_time_remaining > 0.0:
                    if p.kick_mode == "drive_pulse":
                        drive_e += float(p.kick_amp)
                    elif p.kick_mode == "drive_noise":
                        drive_e += float(p.kick_amp) * float(np.random.standard_normal())

                # SOC-friendly stochastic drive: OU (colored, global) + sparse Poisson shots (local, decaying field)
                if p.noise_enabled:
                    if p.ou_enabled:
                        tau = max(1e-6, float(p.ou_tau_s))
                        ou_x += ((float(p.ou_mu_v) - ou_x) / tau) * dt + float(p.ou_sigma_v) * math.sqrt(dt) * float(
                            rng.standard_normal()
                        )
                    if p.shots_enabled:
                        tau_s = max(1e-6, float(p.shots_decay_tau_s))
                        shot_field *= float(math.exp(-dt / tau_s))
                        if float(p.shots_rate_hz) > 0.0:
                            k = int(rng.poisson(float(p.shots_rate_hz) * dt))
                            events_in_frame += k
                            for _ev in range(k):
                                add_shot_event()

                drive_field = drive_e + (ou_x if (p.noise_enabled and p.ou_enabled) else 0.0)

                if p.noise_enabled and p.shots_enabled:
                    P_e = (p.nu_ee * phi_e + p.nu_ei * phi_i + drive_field + shot_field).astype(np.float32)
                else:
                    P_e = (p.nu_ee * phi_e + p.nu_ei * phi_i + drive_field).astype(np.float32)
                P_i = (p.nu_ie * phi_e + p.nu_ii * phi_i + p.drive_i).astype(np.float32)

                # Synaptodendritic filtering (paper Eq 3)
                # V_tt + (alpha+beta)V_t + alpha*beta*V = alpha*beta*P
                ddV_e = (p.alpha * beta) * (P_e - V_e) - (p.alpha + beta) * dV_e
                ddV_i = (p.alpha * beta) * (P_i - V_i) - (p.alpha + beta) * dV_i
                dV_e += dt * ddV_e
                V_e += dt * dV_e
                dV_i += dt * ddV_i
                V_i += dt * dV_i

                # Firing rates (paper Eq 2)
                Q_e = sigmoid(V_e, p.qmax_e, p.theta_e, p.sigma_e)
                Q_i = sigmoid(V_i, p.qmax_i, p.theta_i, p.sigma_i)

                # Wave operator (paper Eq 1)
                lap_e = laplacian_neumann_dispatch(phi_e, p.laplacian_9pt) / (p.dx * p.dx)
                lap_i = (
                    laplacian_neumann_dispatch(phi_i, p.laplacian_9pt) / (p.dx * p.dx)
                    if p.r_i != 0.0
                    else 0.0
                )

                ddphi_e = (
                    (p.gamma_e * p.gamma_e) * (p.r_e * p.r_e) * lap_e
                    - 2.0 * p.gamma_e * dphi_e
                    - (p.gamma_e * p.gamma_e) * phi_e
                    + (p.gamma_e * p.gamma_e) * Q_e
                )
                ddphi_i = (
                    (p.gamma_i * p.gamma_i) * (p.r_i * p.r_i) * lap_i
                    - 2.0 * p.gamma_i * dphi_i
                    - (p.gamma_i * p.gamma_i) * phi_i
                    + (p.gamma_i * p.gamma_i) * Q_i
                )

                dphi_e += dt * ddphi_e
                phi_e += dt * dphi_e
                dphi_i += dt * ddphi_i
                phi_i += dt * dphi_i
                t += dt
                if kick_time_remaining > 0.0:
                    kick_time_remaining = max(0.0, kick_time_remaining - dt)

                if headless_steps_remaining is not None:
                    headless_steps_remaining -= 1
                    if headless_steps_remaining <= 0:
                        running = False
                        break

            # Update events/sec HUD (exponential moving average over frames)
            tau = max(1e-6, float(p.events_hud_ema_tau_s))
            inst = float(events_in_frame) / float(dt_frame)
            a = float(dt_frame / tau)
            if a >= 1.0:
                events_per_s_ema = inst
            else:
                events_per_s_ema = (1.0 - a) * events_per_s_ema + a * inst

        img = render_field_white_on_gray(phi_e, p.wave_gain, p.bg_gray, p.render_deviation)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (p.window_px, p.window_px))
        screen.blit(surf, (0, 0))

        # HUD
        # Rough stability hint: c_eff ~ gamma_e * r_e (from paper's operator scaling)
        c_eff = p.gamma_e * p.r_e
        cfl = (c_eff * dt / p.dx) * math.sqrt(2.0)
        hud1 = (
            f"gamma_e={p.gamma_e:.4g}  r_e={p.r_e:.4g}  qmax_e={p.qmax_e:.4g}  "
            f"nu_ee={p.nu_ee:.4g}  nu_ei={p.nu_ei:.4g}  drive_e={p.drive_e:.4g}"
        )
        hud2 = (
            f"events/s~{events_per_s_ema:.1f}  std(phi_e)={float(np.std(phi_e)):.3g}  "
            f"dt={dt:.3g}  steps={steps}  CFL~{cfl:.3f}   "
        )
        hud3 = (
            f"Space=pause R=reset K=kick N=noise [ ]=gamma -/==qmax ,/.=nu_ee Esc=quit"
        )
        screen.blit(font.render(hud1, True, (235, 235, 235)), (10, 10))
        screen.blit(font.render(hud2, True, (235, 235, 235)), (10, 28))
        screen.blit(font.render(hud3, True, (235, 235, 235)), (10, 46))

        pygame.display.flip()
        if args.save_frames and (frame_idx % max(1, int(args.save_frames_every)) == 0):
            pygame.image.save(screen, os.path.join(args.save_frames, f"frame_{frame_idx:05d}.png"))
        frame_idx += 1
        if not args.headless:
            clock.tick(p.fps)

    if args.headless and args.save_npz:
        np.savez_compressed(
            args.save_npz,
            phi_e=phi_e,
            dphi_e=dphi_e,
            phi_i=phi_i,
            dphi_i=dphi_i,
            V_e=V_e,
            V_i=V_i,
            t=np.array([t], dtype=np.float64),
        )

    pygame.quit()


if __name__ == "__main__":
    main()

