"""Curve optimizer for smooth movement with near-constant speed.

Key upgrade vs previous version:
- Optimizes against a *2D parametric target* (x(τ), y(τ)) instead of assuming y=f(x).
- Includes helpers to sample cubic Beziers and reparameterize by arc length (τ = normalized arc length).
- Keeps your "few triggers" structure: 2 X triggers (ease-in then ease-out) + 2*len(y_exps) Y triggers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import numpy as np


# -------------------- EASINGS (normalized 0..1) --------------------
def ease_in(t: np.ndarray, r: float) -> np.ndarray:
    return t ** r

def ease_out(t: np.ndarray, r: float) -> np.ndarray:
    return 1.0 - (1.0 - t) ** r


# -------------------- FIT y(t) as sum c_i * t^exp_i (no constant) --------------------
def least_squares_fit_poly_no_const(t: np.ndarray, y: np.ndarray, exps: List[float]) -> np.ndarray:
    """Fit y(t) ~= sum_i c_i * t^(exp_i). No constant term since MoveBy uses deltas."""
    A = np.vstack([t ** e for e in exps]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


# -------------------- CUBIC BEZIER HELPERS --------------------
Point = Tuple[float, float]

def cubic_bezier_xy(u: np.ndarray, p0: Point, p1: Point, p2: Point, p3: Point) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate cubic Bezier B(u) for u in [0,1]."""
    u = np.asarray(u, dtype=float)
    s = 1.0 - u
    b0 = s**3
    b1 = 3*s**2*u
    b2 = 3*s*u**2
    b3 = u**3
    x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
    y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
    return x, y

def resample_bezier_by_arclength(
    p0: Point, p1: Point, p2: Point, p3: Point,
    dense: int = 4000,
    out: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (tau, x_tau, y_tau) where tau in [0,1] corresponds to *normalized arc length*.
    This makes tau proportional to distance traveled along the Bezier (ideal constant-speed time parameter).
    """
    u = np.linspace(0.0, 1.0, dense)
    x, y = cubic_bezier_xy(u, p0, p1, p2, p3)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s[-1]) if float(s[-1]) > 1e-12 else 1.0
    tau_of_u = s / L  # monotone in u

    tau = np.linspace(0.0, 1.0, out)
    # invert tau(u) -> u(tau) by interpolation
    u_of_tau = np.interp(tau, tau_of_u, u)
    x_tau, y_tau = cubic_bezier_xy(u_of_tau, p0, p1, p2, p3)
    return tau, x_tau, y_tau


# -------------------- OPTIMIZER MODEL --------------------
@dataclass
class PieceParams:
    # Two time segments: [0, t_split] and [t_split, 1]
    t_split: float
    # Two distance segments in X: [0, x_split] and [x_split, 1]
    x_split: float
    # X easing exponents
    r_in: float
    r_out: float

    # Metrics
    score: float = 1e18
    max_dev: float = 1e18
    rms_dev: float = 1e18
    shape_rmse: float = 1e18

    # Fitted Y coefficients (per segment)
    y_coeffs_1: Optional[np.ndarray] = None
    y_coeffs_2: Optional[np.ndarray] = None


def _implemented_xy_from_params(
    tau: np.ndarray,
    p: PieceParams,
    y_exps: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute implemented (x(tau), y(tau)) in normalized coordinates."""
    tau = np.asarray(tau, dtype=float)
    t_split = float(p.t_split)
    x_split = float(p.x_split)
    Ts1 = t_split
    Ts2 = 1.0 - t_split

    seg1 = tau <= Ts1
    seg2 = ~seg1

    x_impl = np.zeros_like(tau, dtype=float)
    y_impl = np.zeros_like(tau, dtype=float)

    exps = np.array(y_exps, dtype=float)

    if np.any(seg1):
        t1 = tau[seg1] / Ts1 if Ts1 > 1e-12 else np.zeros(np.sum(seg1))
        t1 = np.clip(t1, 0.0, 1.0)
        x_impl[seg1] = x_split * ease_in(t1, p.r_in)
        if p.y_coeffs_1 is not None and len(p.y_coeffs_1) == len(exps):
            y_impl[seg1] = np.sum(p.y_coeffs_1[:, None] * (t1[None, :] ** exps[:, None]), axis=0)

    if np.any(seg2):
        t2 = (tau[seg2] - Ts1) / Ts2 if Ts2 > 1e-12 else np.zeros(np.sum(seg2))
        t2 = np.clip(t2, 0.0, 1.0)
        x_impl[seg2] = x_split + (1.0 - x_split) * ease_out(t2, p.r_out)
        if p.y_coeffs_1 is not None and p.y_coeffs_2 is not None and len(p.y_coeffs_2) == len(exps):
            y_at_split = float(np.sum(p.y_coeffs_1))
            y_impl[seg2] = y_at_split + np.sum(p.y_coeffs_2[:, None] * (t2[None, :] ** exps[:, None]), axis=0)

    return x_impl, y_impl


def build_profile_and_score_targets(
    tau: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    T: float,
    p: PieceParams,
    y_exps: List[float],
) -> PieceParams:
    """Fit Y coefficients and score speed flatness + shape error vs (x_target,y_target)."""
    tau = np.asarray(tau, dtype=float)
    x_target = np.asarray(x_target, dtype=float)
    y_target = np.asarray(y_target, dtype=float)

    Ts1 = float(p.t_split)
    Ts2 = 1.0 - Ts1
    seg1 = tau <= Ts1
    seg2 = ~seg1

    if np.any(seg1):
        t1 = tau[seg1] / Ts1 if Ts1 > 1e-12 else np.zeros(np.sum(seg1))
        t1 = np.clip(t1, 0.0, 1.0)
        y1 = y_target[seg1] - float(y_target[0])
        c1 = least_squares_fit_poly_no_const(t1, y1, y_exps)
    else:
        c1 = np.zeros(len(y_exps), dtype=float)

    if np.any(seg2):
        t2 = (tau[seg2] - Ts1) / Ts2 if Ts2 > 1e-12 else np.zeros(np.sum(seg2))
        t2 = np.clip(t2, 0.0, 1.0)
        y_at_split_target = float(y_target[seg1][-1]) if np.any(seg1) else float(y_target[0])
        y2 = y_target[seg2] - y_at_split_target
        c2 = least_squares_fit_poly_no_const(t2, y2, y_exps)
    else:
        c2 = np.zeros(len(y_exps), dtype=float)

    p.y_coeffs_1 = c1
    p.y_coeffs_2 = c2

    x_impl, y_impl = _implemented_xy_from_params(tau, p, y_exps)

    dist2 = (x_impl - x_target) ** 2 + (y_impl - y_target) ** 2
    shape_rmse = float(np.sqrt(np.mean(dist2)))

    dx = np.diff(x_impl)
    dy = np.diff(y_impl)
    dt = float(T) / max(1, (len(tau) - 1))
    v = np.sqrt(dx * dx + dy * dy) / dt
    if len(v) == 0:
        return p

    v_mean = float(np.mean(v))
    dev = (v / v_mean) - 1.0
    max_dev = float(np.max(np.abs(dev)))
    rms_dev = float(np.sqrt(np.mean(dev ** 2)))

    score = (2.0 * max_dev) + (1.0 * rms_dev) + (3.0 * shape_rmse)

    p.score = score
    p.max_dev = max_dev
    p.rms_dev = rms_dev
    p.shape_rmse = shape_rmse
    return p


def optimize_bezier_profile(
    tau: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    T: float,
    y_exps: List[float],
    iters: int = 4000,
    seed: int = 1,
    t_split_range: Tuple[float, float] = (0.3, 0.7),
    x_split_range: Tuple[float, float] = (0.3, 0.7),
    rate_range: Tuple[float, float] = (0.5, 10.0),
    verbose: bool = True,
) -> PieceParams:
    """Random search over (t_split, x_split, r_in, r_out) with Y coeffs fit per candidate."""
    random.seed(seed)
    best = PieceParams(0.5, 0.5, 2.0, 2.0)
    best = build_profile_and_score_targets(tau, x_target, y_target, T, best, y_exps)

    if verbose:
        print(f"Starting optimization with {iters} iterations...")

    for i in range(iters):
        if verbose and i % 800 == 0 and i > 0:
            print(f"  Iteration {i}/{iters}, best score: {best.score:.6f} | max_dev={best.max_dev:.3f} shape_rmse={best.shape_rmse:.4f}")

        cand = PieceParams(
            t_split=random.uniform(*t_split_range),
            x_split=random.uniform(*x_split_range),
            r_in=random.uniform(*rate_range),
            r_out=random.uniform(*rate_range),
        )
        cand = build_profile_and_score_targets(tau, x_target, y_target, T, cand, y_exps)
        if cand.score < best.score:
            best = cand

    if verbose:
        print(f"Optimization complete. Final score: {best.score:.6f}")
        print(f"  Max deviation: {best.max_dev:.4f} ({best.max_dev*100:.2f}%)")
        print(f"  RMS deviation: {best.rms_dev:.4f} ({best.rms_dev*100:.2f}%)")
        print(f"  Shape RMSE:    {best.shape_rmse:.5f} (normalized units)")

    return best


def implemented_path_samples(
    p: PieceParams,
    y_exps: List[float],
    n: int = 400,
    T: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau, x_impl, y_impl) for plotting/debugging."""
    tau = np.linspace(0.0, 1.0, n)
    x_impl, y_impl = _implemented_xy_from_params(tau, p, y_exps)
    return tau, x_impl, y_impl
