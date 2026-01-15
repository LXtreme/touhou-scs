"""Curve optimizer for smooth movement with near-constant speed."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import random

import numpy as np


def ease_in(t: np.ndarray, r: float) -> np.ndarray:
    return t ** r

def ease_out(t: np.ndarray, r: float) -> np.ndarray:
    return 1.0 - (1.0 - t) ** r

def least_squares_fit_y(t: np.ndarray, y: np.ndarray, exps: List[float]) -> np.ndarray:
    """Fit y(t) ~= sum_i c_i * t^(exp_i). No constant term since MoveBy uses deltas."""
    A = np.vstack([t ** e for e in exps]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


@dataclass
class PieceParams:
    t_split: float      # time split fraction
    x_split: float      # x distance split fraction
    r_in: float         # ease-in exponent for segment 1
    r_out: float        # ease-out exponent for segment 2
    score: float = 1e18
    max_dev: float = 1e18
    rms_dev: float = 1e18
    y_coeffs_1: Optional[np.ndarray] = None
    y_coeffs_2: Optional[np.ndarray] = None


def build_profile_and_score(
    f: Callable[[np.ndarray], np.ndarray],
    x0: float,
    x1: float,
    T: float,
    p: PieceParams,
    y_exps: List[float],
    samples: int = 200
) -> PieceParams:
    """Build movement profile and score its speed consistency."""
    Ts1 = T * p.t_split
    Ts2 = T * (1 - p.t_split)
    x_mid = x0 + (x1 - x0) * p.x_split


    tau1 = np.linspace(0, 1, samples)
    x1_tau = x0 + (x_mid - x0) * ease_in(tau1, p.r_in)
    y1_tau = f(x1_tau)
    y1_0 = f(np.array(x0))
    dy1 = y1_tau - y1_0
    c1 = least_squares_fit_y(tau1, dy1, y_exps)


    tau2 = np.linspace(0, 1, samples)
    x2_tau = x_mid + (x1 - x_mid) * ease_out(tau2, p.r_out)
    y2_tau = f(x2_tau)
    y2_0 = f(np.array(x_mid))
    dy2 = y2_tau - y2_0
    c2 = least_squares_fit_y(tau2, dy2, y_exps)


    exps_arr = np.array(y_exps)
    

    x1_pos = x0 + (x_mid - x0) * ease_in(tau1, p.r_in)
    y1_pos = y1_0 + np.sum(c1[:, None] * (tau1[None, :] ** exps_arr[:, None]), axis=0)
    

    x2_pos = x_mid + (x1 - x_mid) * ease_out(tau2, p.r_out)
    y2_pos = y2_0 + np.sum(c2[:, None] * (tau2[None, :] ** exps_arr[:, None]), axis=0)
    

    dx1_diff = np.diff(x1_pos)
    dy1_diff = np.diff(y1_pos)
    dt1 = Ts1 / (samples - 1)
    v1 = np.sqrt(dx1_diff**2 + dy1_diff**2) / dt1
    
    dx2_diff = np.diff(x2_pos)
    dy2_diff = np.diff(y2_pos)
    dt2 = Ts2 / (samples - 1)
    v2 = np.sqrt(dx2_diff**2 + dy2_diff**2) / dt2
    

    v = np.concatenate([v1, v2])
    
    if len(v) == 0:
        return p

    v_mean = np.mean(v)
    dev = (v / v_mean) - 1.0

    max_dev = float(np.max(np.abs(dev)))
    rms_dev = float(np.sqrt(np.mean(dev ** 2)))


    score = 2.0 * max_dev + 1.0 * rms_dev

    p.score = score
    p.max_dev = max_dev
    p.rms_dev = rms_dev
    p.y_coeffs_1 = c1
    p.y_coeffs_2 = c2
    return p


def optimize_6_trigger_profile(
    f: Callable[[np.ndarray], np.ndarray],
    x0: float,
    x1: float,
    T: float,
    y_exps: List[float],
    iters: int = 4000,
    seed: int = 1,
    t_split_range: Tuple[float, float] = (0.3, 0.7),
    x_split_range: Tuple[float, float] = (0.3, 0.7),
    rate_range: Tuple[float, float] = (0.5, 10.0),
    samples: int = 200
) -> PieceParams:
    """Optimize movement parameters using random search."""
    random.seed(seed)
    best = PieceParams(0.5, 0.5, 2.0, 2.0)
    

    best = build_profile_and_score(f, x0, x1, T, best, y_exps, samples=samples)
    
    print(f"Starting optimization with {iters} iterations...")
    
    for i in range(iters):
        if i % 500 == 0 and i > 0:
            print(f"  Iteration {i}/{iters}, best score: {best.score:.6f}")
        
        cand = PieceParams(
            t_split=random.uniform(*t_split_range),
            x_split=random.uniform(*x_split_range),
            r_in=random.uniform(*rate_range),
            r_out=random.uniform(*rate_range),
        )
        cand = build_profile_and_score(f, x0, x1, T, cand, y_exps, samples=samples)
        if cand.score < best.score:
            best = cand

    print(f"Optimization complete. Final score: {best.score:.6f}")
    print(f"  Max deviation: {best.max_dev:.4f} ({best.max_dev*100:.2f}%)")
    print(f"  RMS deviation: {best.rms_dev:.4f} ({best.rms_dev*100:.2f}%)")
    
    return best


Trigger = Tuple[float, float, float, str, float]

def emit_triggers(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    total_time: float,
    p: PieceParams,
    y_exps: List[float]
) -> List[Trigger]:
    """Convert optimized parameters into trigger list."""
    T = total_time
    Ts1 = T * p.t_split
    Ts2 = T * (1 - p.t_split)
    x_mid = x0 + (x1 - x0) * p.x_split

    dx1 = x_mid - x0
    dx2 = x1 - x_mid

    triggers: List[Trigger] = []


    triggers.append((dx1, 0.0, Ts1, "ease_in", p.r_in))

    if p.y_coeffs_1 is not None:
        for coef, exp in zip(p.y_coeffs_1, y_exps):
            if abs(exp - 1.0) < 1e-9:
                triggers.append((0.0, float(coef), Ts1, "linear", 1.0))
            else:
                triggers.append((0.0, float(coef), Ts1, "ease_in", exp))


    triggers.append((dx2, 0.0, Ts2, "ease_out", p.r_out))

    if p.y_coeffs_2 is not None:
        for coef, exp in zip(p.y_coeffs_2, y_exps):
            if abs(exp - 1.0) < 1e-9:
                triggers.append((0.0, float(coef), Ts2, "linear", 1.0))
            else:
                triggers.append((0.0, float(coef), Ts2, "ease_in", exp))

    return triggers