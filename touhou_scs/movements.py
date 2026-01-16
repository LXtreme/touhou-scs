# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""
Bezier curve movement system for smooth, speed-consistent object motion.

HYBRID OPTIMIZER SYSTEM:
    Automatically selects between OLD and NEW optimizers based on curve complexity:
    - OLD optimizer: 2-segment piecewise (best for extreme curves, tight loops)
    - NEW optimizer: Basis-fit with arc-length (best for gentle curves)
    - Threshold: 0.5 (max control point distance from diagonal)

SYSTEM OVERVIEW:
    1. Define cubic Bezier curves with 4 control points (p0, p1, p2, p3)
    2. Optimization uses either OLD or NEW approach based on curve characteristics
    3. Parameters are stored in CURVE_PARAMS dict and CurveType enum (auto-generated)
    4. At runtime, apply_bezier_movement() generates multiple MoveBy triggers
    5. Optional: Generate animated GIF previews for visual debugging

FILE ORGANIZATION (~850 lines in logical sections):
    - Configuration & Constants       ← QUALITY_PRESETS, ranges, etc.
    - Type Definitions                ← TypedDict for CURVE_PARAMS structure
    - Core Math Functions             ← ease_in, ease_out, curve fitting
    - Curve Fitting & Optimization    ← Random search optimization
    - Auto-Generated Registry         ← CurveType enum + CURVE_PARAMS dict
    - Registration System             ← register_bezier_curve() + self-modifying code
    - Runtime Application             ← apply_bezier_movement() used by Component
    - Preview Generation (Optional)   ← matplotlib GIF generation (lazy import)

SETUP:
    python setup_curves.py          # Register all common curves (one-time setup)
    python generate_all_previews.py # Generate GIFs for all curves (optional)

USAGE:
    from touhou_scs.movements import CurveType
    component.timed.BezierMove(0.5, CurveType.BOSS_CHARGE, dx=100, dy=-50, t=3.0)
    
DEVELOPER EXPERIENCE NOTE:
    The auto-generated CurveType enum provides autocomplete and type safety.
    Yes, the code modifies itself - this is intentional for DX! :)
"""

from __future__ import annotations
from typing import Union, TYPE_CHECKING, List, Any, Callable, Optional, Tuple, TypedDict, Dict, Literal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
import json
import hashlib
import numpy as np
from multiprocessing import Pool, Process, cpu_count, get_context
from touhou_scs import enums as e

if TYPE_CHECKING: from touhou_scs.component import Component

# Queue for preview generation specs
_preview_queue: List[Tuple[str, CurveParams, float, float, float]] = []


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Optimization settings
OPTIMIZATION_SAMPLES = 200  # Sample points for curve fitting

# Quality presets define optimization iterations, polynomial complexity, and trigger count
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "fast":   {"iters": 1000, "y_exps": [1.0, 2.0],             "triggers": 6},
    "medium": {"iters": 1000, "y_exps": [1.0, 2.0, 3.0],        "triggers": 8},
    "high":   {"iters": 4000, "y_exps": [1.0, 2.0, 3.0],        "triggers": 8},
    "ultra":  {"iters": 4000, "y_exps": [1.0, 2.0, 3.0, 4.0],   "triggers": 10},
}

# Preview generation settings
PREVIEW_FPS = 40
PREVIEW_FRAMES = 120
PREVIEW_SAMPLES = 100
PREVIEW_PADDING = 0.15

# Optimization search ranges
SPLIT_TIME_RANGE = (0.3, 0.7)  # Where to split the curve (30%-70% of duration)
SPLIT_DISTANCE_RANGE = (0.3, 0.7)  # Where to split X distance
EASING_RATE_RANGE = (0.5, 10.0)  # Ease-in/ease-out exponent range

# Hybrid optimizer selection
HYBRID_THRESHOLD = 0.5  # Control point distance from start-end line (based on 1000 curve analysis)

# NEW optimizer settings (basis-fit)
TARGET_DENSE = 4000     # for arc-length build
TARGET_OUT = 600        # target samples used for fitting
EVAL_SAMPLES = 500
_OPTIMIZER_VERSION = "basisfit_v1"

# Score weighting for optimizer quality metrics
SCORE_WEIGHT_MAX_DEV = 2.0
SCORE_WEIGHT_RMS_DEV = 1.0
SCORE_WEIGHT_SHAPE = 3.0

# Basis function candidates
DEFAULT_POWER_RATES = [0.65, 0.85, 1.0, 1.25, 1.6, 2.0, 2.8, 3.8, 5.0]
DEFAULT_POWER_RATES_ULTRA = [0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.9, 2.5, 3.2, 4.2, 5.5, 7.5]

# NEW optimizer quality presets
NEW_QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast":   {"x_terms": 3, "y_terms": 3, "power_rates": DEFAULT_POWER_RATES,       "candidates": "basic"},
    "medium": {"x_terms": 4, "y_terms": 4, "power_rates": DEFAULT_POWER_RATES,       "candidates": "basic"},
    "high":   {"x_terms": 4, "y_terms": 4, "power_rates": DEFAULT_POWER_RATES,       "candidates": "rich"},
    "ultra":  {"x_terms": 5, "y_terms": 5, "power_rates": DEFAULT_POWER_RATES_ULTRA, "candidates": "rich"},
}

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

Point = tuple[float | int, float | int]


class BezierPointsDict(TypedDict):
    p0: list[float]
    p1: list[float]
    p2: list[float]
    p3: list[float]


class BasisTermDict(TypedDict):
    kind: str      # "linear" | "ease_in" | "ease_out" | "sine_in" | "sine_out" | etc.
    rate: float    # for power kinds
    coef: float    # weight in mixture


class QualityInfoDict(TypedDict):
    max_speed_dev: float
    rms_speed_dev: float
    shape_rmse: float
    score: float
    quality_preset: str
    trigger_count: int


class QualityInfoDictNew(QualityInfoDict):
    optimizer_version: str
    curve_hash: str


# OLD optimizer parameters (2-segment piecewise)
class OldCurveParams(TypedDict):
    new_system: Literal[False]
    t_split: float
    x_split: float
    r_in: float
    r_out: float
    y_coeffs_1: list[float]
    y_coeffs_2: list[float]
    y_exps: list[float]
    bezier_points: BezierPointsDict
    quality_info: QualityInfoDict


# NEW optimizer parameters (basis-fit)
class NewCurveParams(TypedDict):
    new_system: Literal[True]
    x_terms: list[BasisTermDict]
    y_terms: list[BasisTermDict]
    bezier_points: BezierPointsDict
    quality_info: QualityInfoDictNew


# Union type for all curve parameters
CurveParams = Union[OldCurveParams, NewCurveParams]


@dataclass
class OptimizationResult:
    t_split: float
    x_split: float
    r_in: float
    r_out: float
    score: float = 1e18
    max_speed_dev: float = 1e18
    rms_speed_dev: float = 1e18
    y_coeffs_1: Optional[np.ndarray] = None
    y_coeffs_2: Optional[np.ndarray] = None


# ============================================================================
# CORE MATH FUNCTIONS
# ============================================================================

# OLD optimizer easing functions
def ease_in(t: np.ndarray, rate: float) -> np.ndarray:
    return t**rate


def ease_out(t: np.ndarray, rate: float) -> np.ndarray:
    return 1.0 - (1.0 - t) ** rate


# ============================================================================
# NEW OPTIMIZER - BASIS-FIT FUNCTIONS
# ============================================================================

def _ease_linear(t: np.ndarray) -> np.ndarray:
    return t


def _ease_in_basis(t: np.ndarray, r: float) -> np.ndarray:
    return np.power(t, r)


def _ease_out_basis(t: np.ndarray, r: float) -> np.ndarray:
    return 1.0 - np.power(1.0 - t, r)


def _sine_in(t: np.ndarray) -> np.ndarray:
    return 1.0 - np.cos((np.pi / 2.0) * t)


def _sine_out(t: np.ndarray) -> np.ndarray:
    return np.sin((np.pi / 2.0) * t)


def _sine_inout(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * t))


def _expo_in(t: np.ndarray) -> np.ndarray:
    out = np.where(t <= 0, 0.0, np.power(2.0, 10.0 * (t - 1.0)))
    return out


def _expo_out(t: np.ndarray) -> np.ndarray:
    out = np.where(t >= 1, 1.0, 1.0 - np.power(2.0, -10.0 * t))
    return out


def basis_curve(t: np.ndarray, kind: str, rate: float) -> np.ndarray:
    """Evaluate a monotone easing basis function."""
    t = np.asarray(t, dtype=float)
    if kind == "linear":
        return _ease_linear(t)
    if kind == "ease_in":
        return _ease_in_basis(t, rate)
    if kind == "ease_out":
        return _ease_out_basis(t, rate)
    if kind == "sine_in":
        return _sine_in(t)
    if kind == "sine_out":
        return _sine_out(t)
    if kind == "sine_inout":
        return _sine_inout(t)
    if kind == "expo_in":
        return _expo_in(t)
    if kind == "expo_out":
        return _expo_out(t)
    raise ValueError(f"Unknown basis kind: {kind}")


def term_to_gd_easing(kind: str, rate: float) -> Tuple[Any, float]:
    """Map basis term to GD easing enum + rate."""
    if kind == "linear":
        return e.Easing.NONE, 1.0
    if kind == "ease_in":
        return e.Easing.EASE_IN, float(rate)
    if kind == "ease_out":
        return e.Easing.EASE_OUT, float(rate)
    if kind == "sine_in":
        return e.Easing.SINE_IN, 1.0
    if kind == "sine_out":
        return e.Easing.SINE_OUT, 1.0
    if kind == "sine_inout":
        return e.Easing.SINE_IN_OUT, 1.0
    if kind == "expo_in":
        return e.Easing.EXPONENTIAL_IN, 1.0
    if kind == "expo_out":
        return e.Easing.EXPONENTIAL_OUT, 1.0
    raise ValueError(f"Unknown kind for GD mapping: {kind}")


def cubic_bezier_xy(u: np.ndarray, p0: Point, p1: Point, p2: Point, p3: Point) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate cubic Bezier at parameter u, returning (x, y)."""
    u = np.asarray(u, dtype=float)
    s = 1.0 - u
    b0 = s**3
    b1 = 3*s**2*u
    b2 = 3*s*u**2
    b3 = u**3
    x = b0*p0[0] + b1*p1[0] + b2*p2[0] + b3*p3[0]
    y = b0*p0[1] + b1*p1[1] + b2*p2[1] + b3*p3[1]
    return x, y


def resample_bezier_by_arclength(p0: Point, p1: Point, p2: Point, p3: Point,
                                 dense: int = TARGET_DENSE, out: int = TARGET_OUT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (t, x(t), y(t)) where t in [0,1] corresponds to normalized arc length.
    This is the ideal constant-speed parameterization along the curve.
    """
    u = np.linspace(0.0, 1.0, dense)
    x, y = cubic_bezier_xy(u, p0, p1, p2, p3)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s[-1]) if float(s[-1]) > 1e-12 else 1.0
    t_of_u = s / L
    t = np.linspace(0.0, 1.0, out)
    u_of_t = np.interp(t, t_of_u, u)
    xt, yt = cubic_bezier_xy(u_of_t, p0, p1, p2, p3)
    return t, xt, yt


def _nnls_with_sum_constraint(G: np.ndarray, y: np.ndarray, w_sum: float = 50.0) -> np.ndarray:
    """
    Solve min ||G a - y||^2 with a>=0 and (sum a = 1) softly enforced.
    Fast iterative solver for small basis sizes.
    """
    n = G.shape[1]
    Gc = np.vstack([G, (w_sum * np.ones((1, n)))])
    yc = np.concatenate([y, np.array([w_sum], dtype=float)])

    active = np.ones(n, dtype=bool)
    a = np.zeros(n, dtype=float)

    for _ in range(10):
        if not np.any(active):
            break
        Ga = Gc[:, active]
        aa, *_ = np.linalg.lstsq(Ga, yc, rcond=None)
        a[:] = 0.0
        a[active] = aa
        neg = (a < -1e-10)
        if not np.any(neg):
            break
        idx = int(np.argmin(a))
        active[idx] = False

    a = np.maximum(a, 0.0)
    s = float(np.sum(a))
    if s > 1e-12:
        a /= s
    else:
        a[:] = 0.0
        a[0] = 1.0
    return a


@dataclass
class FitResult:
    terms: List[BasisTermDict]
    rmse: float


def build_candidates(power_rates: List[float], richness: str) -> List[Tuple[str, float]]:
    """Build candidate basis set for greedy selection."""
    cands: List[Tuple[str, float]] = [("linear", 1.0)]
    for r in power_rates:
        cands.append(("ease_in", float(r)))
        cands.append(("ease_out", float(r)))
    if richness == "rich":
        cands += [("sine_in", 1.0), ("sine_out", 1.0), ("sine_inout", 1.0), ("expo_in", 1.0), ("expo_out", 1.0)]
    return cands


def greedy_fit(target: np.ndarray, t: np.ndarray, candidates: List[Tuple[str, float]], k: int) -> FitResult:
    """
    Greedy forward selection of k basis terms (nonnegative mixture).
    Returns selected term descriptors + coefficients.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(target, dtype=float)

    selected: List[Tuple[str, float]] = []
    remaining = candidates.copy()

    # Ensure linear always present if available
    linear_term = ("linear", 1.0)
    if linear_term in candidates:
        selected.append(linear_term)
        if linear_term in remaining:
            remaining.remove(linear_term)

    def fit_with_set(sel: List[Tuple[str, float]]) -> Tuple[np.ndarray, float]:
        G = np.column_stack([basis_curve(t, kind, rate) for (kind, rate) in sel])
        a = _nnls_with_sum_constraint(G, y, w_sum=60.0)
        pred = G @ a
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return a, rmse

    # Grow until k
    while len(selected) < k and remaining:
        best_rmse = 1e18
        best_term: Optional[Tuple[str, float]] = None
        best_a: Optional[np.ndarray] = None

        for term in remaining:
            sel_try = selected + [term]
            a_try, rmse_try = fit_with_set(sel_try)
            if rmse_try < best_rmse:
                best_rmse = rmse_try
                best_term = term
                best_a = a_try

        if best_term is None or best_a is None:
            break

        selected.append(best_term)
        remaining.remove(best_term)

    # Final fit with selected
    a, rmse = fit_with_set(selected)

    terms_out: List[BasisTermDict] = []
    for (kind, rate), coef in zip(selected, a):
        terms_out.append({"kind": kind, "rate": float(rate), "coef": float(coef)})

    # Drop tiny terms
    terms_out = [t for t in terms_out if abs(t["coef"]) > 1e-6]

    # Renormalize
    s = sum(t["coef"] for t in terms_out)
    if s > 1e-12:
        for tt in terms_out:
            tt["coef"] /= s

    return FitResult(terms=terms_out, rmse=rmse)


def eval_path(xt: np.ndarray, yt: np.ndarray, T: float) -> Tuple[float, float]:
    """Return (max_dev, rms_dev) of speed vs mean speed."""
    pos = np.column_stack([xt, yt])
    dx = np.diff(pos[:, 0])
    dy = np.diff(pos[:, 1])
    dt = float(T) / max(1, (len(pos) - 1))
    v = np.sqrt(dx*dx + dy*dy) / dt
    if len(v) == 0:
        return 1e18, 1e18
    m = float(np.mean(v))
    dev = (v / m) - 1.0
    return float(np.max(np.abs(dev))), float(np.sqrt(np.mean(dev**2)))


def build_implemented_xy(t: np.ndarray, x_terms: List[BasisTermDict], y_terms: List[BasisTermDict]) -> Tuple[np.ndarray, np.ndarray]:
    """Build X and Y paths from basis term mixtures."""
    x = np.zeros_like(t, dtype=float)
    y = np.zeros_like(t, dtype=float)
    for term in x_terms:
        x += float(term["coef"]) * basis_curve(t, term["kind"], float(term["rate"]))
    for term in y_terms:
        y += float(term["coef"]) * basis_curve(t, term["kind"], float(term["rate"]))
    return x, y


def score_fit(t: np.ndarray,
              x_target: np.ndarray, y_target: np.ndarray,
              x_terms: List[BasisTermDict], y_terms: List[BasisTermDict]) -> Tuple[float, float, float, float]:
    """Returns (score, max_dev, rms_dev, shape_rmse)."""
    xi, yi = build_implemented_xy(t, x_terms, y_terms)
    shape_rmse = float(np.sqrt(np.mean((xi - x_target)**2 + (yi - y_target)**2)))
    max_dev, rms_dev = eval_path(xi, yi, T=1.0)
    score = (SCORE_WEIGHT_MAX_DEV * max_dev + 
             SCORE_WEIGHT_RMS_DEV * rms_dev + 
             SCORE_WEIGHT_SHAPE * shape_rmse)
    return score, max_dev, rms_dev, shape_rmse


def _hash_curve(p0: Point, p1: Point, p2: Point, p3: Point, quality: str) -> str:
    """Generate hash for curve identity."""
    payload = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "quality": quality, "v": _OPTIMIZER_VERSION}
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def register_bezier_curve_new(p0: Point, p1: Point, p2: Point, p3: Point,
                               quality: str = "medium") -> NewCurveParams:
    """
    NEW optimizer: Register using basis-fit approach.
    Returns params dict for storage in CURVE_PARAMS.
    """
    if quality not in NEW_QUALITY_PRESETS:
        raise ValueError(f"quality must be one of {list(NEW_QUALITY_PRESETS.keys())}, got '{quality}'")

    curve_hash = _hash_curve(p0, p1, p2, p3, quality)

    settings = NEW_QUALITY_PRESETS[quality]
    x_budget = int(settings["x_terms"])
    y_budget = int(settings["y_terms"])
    power_rates: List[float] = list(settings["power_rates"])
    richness = str(settings["candidates"])

    print(f"  NEW optimizer: X={x_budget} terms, Y={y_budget} terms")

    # Build constant-speed target
    t, xt, yt = resample_bezier_by_arclength(p0, p1, p2, p3, dense=TARGET_DENSE, out=TARGET_OUT)

    # Normalize to (0..1) displacement space
    x0 = float(xt[0])
    x1 = float(xt[-1])
    y0 = float(yt[0])
    y1 = float(yt[-1])

    dx = (x1 - x0) if abs(x1 - x0) > 1e-12 else 1.0
    dy = (y1 - y0) if abs(y1 - y0) > 1e-12 else 1.0

    x_target = (xt - x0) / dx
    y_target = (yt - y0) / dy

    # Greedy fit
    candidates = build_candidates(power_rates, richness)
    x_fit = greedy_fit(x_target, t, candidates, k=x_budget)
    y_fit = greedy_fit(y_target, t, candidates, k=y_budget)

    # Score
    score, max_dev, rms_dev, shape_rmse = score_fit(t, x_target, y_target, x_fit.terms, y_fit.terms)

    params: NewCurveParams = {
        "new_system": True,
        "x_terms": x_fit.terms,
        "y_terms": y_fit.terms,
        "bezier_points": {"p0": [float(p0[0]), float(p0[1])],
                          "p1": [float(p1[0]), float(p1[1])],
                          "p2": [float(p2[0]), float(p2[1])],
                          "p3": [float(p3[0]), float(p3[1])]},
        "quality_info": {
            "max_speed_dev": float(max_dev),
            "rms_speed_dev": float(rms_dev),
            "shape_rmse": float(shape_rmse),
            "score": float(score),
            "quality_preset": quality,
            "trigger_count": len(x_fit.terms) + len(y_fit.terms),
            "optimizer_version": _OPTIMIZER_VERSION,
            "curve_hash": curve_hash,
        }
    }

    trigger_count = len(x_fit.terms) + len(y_fit.terms)
    print(f"  Triggers: {trigger_count}  |  Max speed dev: {max_dev*100:.2f}%  |  Shape RMSE: {shape_rmse:.5f}")
    return params


def fit_polynomial_least_squares(
    t: np.ndarray, y: np.ndarray, exponents: List[float]
) -> np.ndarray:
    """Fit y(t) = sum(c_i * t^exp_i) using least squares. No constant term."""
    A = np.vstack([t**exp for exp in exponents]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def compute_speed_profile(positions: np.ndarray, dt: float) -> np.ndarray:
    """Calculate instantaneous speed from position samples."""
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / dt
    return speeds


def score_speed_consistency(speeds: np.ndarray) -> Tuple[float, float, float]:
    """Returns (score, max_deviation, rms_deviation). Lower is better."""
    if len(speeds) == 0:
        return 1e18, 1e18, 1e18

    mean_speed = np.mean(speeds)
    fractional_deviation = (speeds / mean_speed) - 1.0

    max_dev = float(np.max(np.abs(fractional_deviation)))
    rms_dev = float(np.sqrt(np.mean(fractional_deviation**2)))

    score = 2.0 * max_dev + 1.0 * rms_dev  # Weight max deviation more heavily
    return score, max_dev, rms_dev


# ============================================================================
# CURVE FITTING & OPTIMIZATION
# ============================================================================
#
# WHY 2-SEGMENT PIECEWISE APPROACH?
# 
# Problem: Bezier curves don't have constant speed - objects accelerate/decelerate
# along the curve, causing jerky motion in GD.
#
# Solution: Split the curve into 2 segments with:
#   - Segment 1: Ease-in X motion + polynomial Y motion (0% to split%)
#   - Segment 2: Ease-out X motion + polynomial Y motion (split% to 100%)
#
# This gives us enough control points to approximate the Bezier curve while
# maintaining relatively constant speed. The optimization finds the best:
#   - Split point (where to divide the curve)
#   - Easing rates (how aggressive the ease-in/out should be)
#   - Polynomial coefficients (how Y changes over time)
#
# Result: Smooth curves with <50% speed deviation (most <25%)
# ============================================================================


def fit_and_evaluate_curve(
    bezier_func: Callable[[np.ndarray], np.ndarray],
    x0: float,
    x1: float,
    total_duration: float,
    params: OptimizationResult,
    y_exponents: List[float],
) -> OptimizationResult:
    """Fit 2-segment piecewise curve and score speed consistency."""
    duration_seg1 = total_duration * params.t_split
    duration_seg2 = total_duration * (1 - params.t_split)
    x_mid = x0 + (x1 - x0) * params.x_split

    # Segment 1: Ease-in
    tau1 = np.linspace(0, 1, OPTIMIZATION_SAMPLES)
    x1_positions = x0 + (x_mid - x0) * ease_in(tau1, params.r_in)
    y1_positions = bezier_func(x1_positions)
    y1_start = bezier_func(np.array(x0))
    dy1 = y1_positions - y1_start
    coeffs1 = fit_polynomial_least_squares(tau1, dy1, y_exponents)

    # Segment 2: Ease-out
    tau2 = np.linspace(0, 1, OPTIMIZATION_SAMPLES)
    x2_positions = x_mid + (x1 - x_mid) * ease_out(tau2, params.r_out)
    y2_positions = bezier_func(x2_positions)
    y2_start = bezier_func(np.array(x_mid))
    dy2 = y2_positions - y2_start
    coeffs2 = fit_polynomial_least_squares(tau2, dy2, y_exponents)

    # Reconstruct full path
    exps_arr = np.array(y_exponents)
    x1_reconstructed = x0 + (x_mid - x0) * ease_in(tau1, params.r_in)
    y1_reconstructed = y1_start + np.sum(
        coeffs1[:, None] * (tau1[None, :] ** exps_arr[:, None]), axis=0
    )
    x2_reconstructed = x_mid + (x1 - x_mid) * ease_out(tau2, params.r_out)
    y2_reconstructed = y2_start + np.sum(
        coeffs2[:, None] * (tau2[None, :] ** exps_arr[:, None]), axis=0
    )

    positions_seg1 = np.column_stack([x1_reconstructed, y1_reconstructed])
    positions_seg2 = np.column_stack([x2_reconstructed, y2_reconstructed])

    # Compute speed profile
    dt1 = duration_seg1 / (OPTIMIZATION_SAMPLES - 1)
    dt2 = duration_seg2 / (OPTIMIZATION_SAMPLES - 1)
    speeds1 = compute_speed_profile(positions_seg1, dt1)
    speeds2 = compute_speed_profile(positions_seg2, dt2)
    all_speeds = np.concatenate([speeds1, speeds2])
    score, max_dev, rms_dev = score_speed_consistency(all_speeds)

    params.score = score
    params.max_speed_dev = max_dev
    params.rms_speed_dev = rms_dev
    params.y_coeffs_1 = coeffs1
    params.y_coeffs_2 = coeffs2

    return params


def optimize_curve_parameters(
    bezier_func: Callable[[np.ndarray], np.ndarray],
    x0: float,
    x1: float,
    total_duration: float,
    y_exponents: List[float],
    iterations: int = 1000,
    seed: int = 1,
) -> OptimizationResult:
    """Find optimal piecewise curve parameters using random search."""
    random.seed(seed)

    best = OptimizationResult(t_split=0.5, x_split=0.5, r_in=2.0, r_out=2.0)
    best = fit_and_evaluate_curve(bezier_func, x0, x1, total_duration, best, y_exponents)

    print(f"Optimizing with {iterations} iterations...")

    for i in range(iterations):
        if i % 500 == 0 and i > 0:
            print(f"  Iteration {i}/{iterations}, best score: {best.score:.6f}")

        candidate = OptimizationResult(
            t_split=random.uniform(*SPLIT_TIME_RANGE),
            x_split=random.uniform(*SPLIT_DISTANCE_RANGE),
            r_in=random.uniform(*EASING_RATE_RANGE),
            r_out=random.uniform(*EASING_RATE_RANGE),
        )

        candidate = fit_and_evaluate_curve(
            bezier_func, x0, x1, total_duration, candidate, y_exponents
        )

        if candidate.score < best.score:
            best = candidate

    print(f"Optimization complete. Final score: {best.score:.6f}")
    print(f"  Max speed deviation: {best.max_speed_dev:.4f} ({best.max_speed_dev*100:.2f}%)")
    print(f"  RMS speed deviation: {best.rms_speed_dev:.4f} ({best.rms_speed_dev*100:.2f}%)")

    return best


# ============================================================================
# AUTO-GENERATED SECTION - DO NOT EDIT BELOW THIS LINE

class CurveType(Enum):
    BOSS_CHARGE = "boss_charge"
    BOSS_WEAVE = "boss_weave"
    FAST_ARC = "fast_arc"
    GENTLE_ARC = "gentle_arc"
    GENTLE_ARC_DOWN = "gentle_arc_down"
    S_CURVE = "s_curve"
    S_CURVE_REVERSE = "s_curve_reverse"
    SMOOTH_EASE = "smooth_ease"
    STEEP_DIVE = "steep_dive"
    STEEP_RISE = "steep_rise"

CURVE_PARAMS = {
    "boss_charge": {
        "new_system": True,
        "x_terms": [{'kind': 'linear', 'rate': 1.0, 'coef': 0.7725710252314457}, {'kind': 'ease_in', 'rate': 3.2, 'coef': 0.07170506753064487}, {'kind': 'ease_in', 'rate': 0.55, 'coef': 0.03362057734898485}, {'kind': 'ease_in', 'rate': 2.5, 'coef': 0.12035124310217479}, {'kind': 'ease_in', 'rate': 7.5, 'coef': 0.001752086786749802}],
        "y_terms": [{'kind': 'ease_out', 'rate': 1.2, 'coef': 0.6252834774593973}, {'kind': 'sine_inout', 'rate': 1.0, 'coef': 0.02297110176849344}, {'kind': 'sine_out', 'rate': 1.0, 'coef': 0.17783723468879556}, {'kind': 'ease_in', 'rate': 1.2, 'coef': 0.17390818608331376}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.1, 0.1], 'p2': [0.5, 0.8], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.2173047021191421, 'rms_speed_dev': 0.009951398146932935, 'shape_rmse': 0.0004461025844472912, 'score': 0.445899110138559, 'quality_preset': 'ultra', 'trigger_count': 9, 'optimizer_version': 'basisfit_v1', 'curve_hash': '99ab92d7f1edfdb4'}
    },
    "boss_weave": {
        "new_system": False,
        "t_split": 0.3829191101399722, "x_split": 0.36300775258780243, "r_in": 1.3909276295329414, "r_out": 1.45263516774201,
        "y_coeffs_1": [0.3980398804582759, 1.633353111065155, -2.402945826623945, 0.9018260459046324],
        "y_coeffs_2": [0.014232771353400073, -1.9712574236536344, 5.190751084637478, -2.754041061343797],
        "y_exps": [1.0, 2.0, 3.0, 4.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.25, 1.2], 'p2': [0.75, -0.2], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.3772276661640659, 'rms_speed_dev': 0.20836393677836307, 'shape_rmse': 0.0, 'score': 0.9628192691064948, 'quality_preset': 'ultra', 'trigger_count': 10}
    },
    "fast_arc": {
        "new_system": True,
        "x_terms": [{'kind': 'ease_in', 'rate': 1.9, 'coef': 0.5328545837461446}, {'kind': 'ease_out', 'rate': 1.2, 'coef': 0.4127518236254053}, {'kind': 'ease_in', 'rate': 5.5, 'coef': 0.0543935926284502}],
        "y_terms": [{'kind': 'sine_inout', 'rate': 1.0, 'coef': 0.45906088279562135}, {'kind': 'ease_out', 'rate': 2.5, 'coef': 0.5409391172043787}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.2, 0.6], 'p2': [0.8, 1.2], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.06576949263745135, 'rms_speed_dev': 0.0426475194783836, 'shape_rmse': 0.019021282631344893, 'score': 0.231250352647321, 'quality_preset': 'ultra', 'trigger_count': 5, 'optimizer_version': 'basisfit_v1', 'curve_hash': 'f7d69481d4b9e6a6'}
    },
    "gentle_arc": {
        "new_system": True,
        "x_terms": [{'kind': 'ease_in', 'rate': 1.9, 'coef': 0.5328545837461446}, {'kind': 'ease_out', 'rate': 1.2, 'coef': 0.4127518236254053}, {'kind': 'ease_in', 'rate': 5.5, 'coef': 0.0543935926284502}],
        "y_terms": [{'kind': 'sine_inout', 'rate': 1.0, 'coef': 0.45906088279562135}, {'kind': 'ease_out', 'rate': 2.5, 'coef': 0.5409391172043787}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.2, 0.6], 'p2': [0.8, 1.2], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.06576949263745135, 'rms_speed_dev': 0.0426475194783836, 'shape_rmse': 0.019021282631344893, 'score': 0.231250352647321, 'quality_preset': 'ultra', 'trigger_count': 5, 'optimizer_version': 'basisfit_v1', 'curve_hash': 'f7d69481d4b9e6a6'}
    },
    "gentle_arc_down": {
        "new_system": True,
        "x_terms": [{'kind': 'ease_out', 'rate': 4.2, 'coef': 0.13916096829550936}, {'kind': 'ease_out', 'rate': 3.2, 'coef': 0.1286565706566457}, {'kind': 'ease_in', 'rate': 1.2, 'coef': 0.6985601353916547}, {'kind': 'ease_out', 'rate': 1.5, 'coef': 0.0336223256561902}],
        "y_terms": [{'kind': 'sine_inout', 'rate': 1.0, 'coef': 0.6851231050319321}, {'kind': 'ease_in', 'rate': 4.2, 'coef': 0.3148768949680679}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.2, -0.3], 'p2': [0.8, 0.7], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.15937451800441904, 'rms_speed_dev': 0.05361398859757059, 'shape_rmse': 0.025701020509125785, 'score': 0.449466086133786, 'quality_preset': 'ultra', 'trigger_count': 6, 'optimizer_version': 'basisfit_v1', 'curve_hash': 'c6386061202ecaa8'}
    },
    "s_curve": {
        "new_system": False,
        "t_split": 0.3829191101399722, "x_split": 0.36300775258780243, "r_in": 1.3909276295329414, "r_out": 1.45263516774201,
        "y_coeffs_1": [0.4982060560785452, 2.0301517657270396, -3.1074159413568, 1.1670003973031249],
        "y_coeffs_2": [-0.3046813504966531, -2.5533331876596934, 6.936334290251212, -3.653893569141854],
        "y_exps": [1.0, 2.0, 3.0, 4.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, 1.5], 'p2': [0.8, -0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.4655699535946667, 'rms_speed_dev': 0.2628441678832258, 'shape_rmse': 0.0, 'score': 1.1939840750725592, 'quality_preset': 'ultra', 'trigger_count': 10}
    },
    "s_curve_reverse": {
        "new_system": True,
        "x_terms": [{'kind': 'ease_out', 'rate': 5.5, 'coef': 0.19783311736015793}, {'kind': 'ease_in', 'rate': 5.5, 'coef': 0.1062739106058945}, {'kind': 'ease_in', 'rate': 1.2, 'coef': 0.6958929720339476}],
        "y_terms": [{'kind': 'sine_inout', 'rate': 1.0, 'coef': 1.0}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.2, -0.5], 'p2': [0.8, 1.5], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.17866829678271157, 'rms_speed_dev': 0.11559723667042056, 'shape_rmse': 0.045270323802165266, 'score': 0.6087448016423395, 'quality_preset': 'ultra', 'trigger_count': 4, 'optimizer_version': 'basisfit_v1', 'curve_hash': 'c030e3a7735ae6db'}
    },
    "smooth_ease": {
        "new_system": True,
        "x_terms": [{'kind': 'linear', 'rate': 1.0, 'coef': 0.4835592865969358}, {'kind': 'ease_in', 'rate': 7.5, 'coef': 0.0812484689230947}, {'kind': 'ease_out', 'rate': 7.5, 'coef': 0.09512634358090787}, {'kind': 'ease_in', 'rate': 5.5, 'coef': 0.0529380173203246}, {'kind': 'ease_out', 'rate': 1.2, 'coef': 0.287127883578737}],
        "y_terms": [{'kind': 'sine_inout', 'rate': 1.0, 'coef': 0.43923575729721487}, {'kind': 'ease_out', 'rate': 1.2, 'coef': 0.2796408974211905}, {'kind': 'ease_in', 'rate': 1.2, 'coef': 0.28112334528159455}],
        "bezier_points": {'p0': [0.0, 0.0], 'p1': [0.33, 0.0], 'p2': [0.67, 1.0], 'p3': [1.0, 1.0]},
        "quality_info": {'max_speed_dev': 0.10040548764367063, 'rms_speed_dev': 0.03681811404666922, 'shape_rmse': 0.007009969576997846, 'score': 0.25865899806500403, 'quality_preset': 'ultra', 'trigger_count': 8, 'optimizer_version': 'basisfit_v1', 'curve_hash': '194070cd33ae855c'}
    },
    "steep_dive": {
        "new_system": False,
        "t_split": 0.3838947235978839, "x_split": 0.3804607334987112, "r_in": 0.9297341708340849, "r_out": 1.022084549655822,
        "y_coeffs_1": [-0.6732859821092486, 1.0640593055026195, -0.6221728020002595, 0.20296983759590836],
        "y_coeffs_2": [0.6698180741795261, 0.8718163587536909, -0.5098679093035038, -0.0016762089581020545],
        "y_exps": [1.0, 2.0, 3.0, 4.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.3, -0.5], 'p2': [0.7, 0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.42160757600512433, 'rms_speed_dev': 0.2736589155300678, 'shape_rmse': 0.0, 'score': 1.1168740675403166, 'quality_preset': 'ultra', 'trigger_count': 10}
    },
    "steep_rise": {
        "new_system": False,
        "t_split": 0.5429342019430756, "x_split": 0.4317749206381954, "r_in": 1.1469689285636413, "r_out": 1.2848995828571697,
        "y_coeffs_1": [1.3054778692643012, 0.6175966290604433, -1.8945708415128544, 0.8418776933609704],
        "y_coeffs_2": [0.21429344179775622, -1.4567454108630788, 2.3689013863494393, -0.9883494029662958],
        "y_exps": [1.0, 2.0, 3.0, 4.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.3, 1.5], 'p2': [0.7, 0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.6205096477883609, 'rms_speed_dev': 0.3066047528955823, 'shape_rmse': 0.0, 'score': 1.5476240484723043, 'quality_preset': 'ultra', 'trigger_count': 10}
    },
}

# END AUTO-GENERATED SECTION
# ============================================================================

# Type annotation for the auto-generated CURVE_PARAMS
CURVE_PARAMS: Dict[str, CurveParams]


# ============================================================================
# CURVE REGISTRATION
# ============================================================================


def _point_to_line_distance(p: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line segment."""
    line_vec = np.array([end[0] - start[0], end[1] - start[1]])
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return 0.0
    line_unit = line_vec / line_len
    
    point_vec = np.array([p[0] - start[0], p[1] - start[1]])
    proj_len = np.dot(point_vec, line_unit)
    proj_point = np.array(start) + proj_len * line_unit
    
    dist = np.linalg.norm(point_vec - (proj_point - np.array(start)))
    return float(dist)


def calculate_curve_distance(p0: Point, p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate maximum control point distance from start-end line.
    This metric indicates curve complexity:
    - Low distance (<0.5): Gentle arcs, easy to optimize
    - High distance (>0.5): Tight loops/S-curves, harder to optimize
    """
    dist_p1 = _point_to_line_distance(p1, p0, p3)
    dist_p2 = _point_to_line_distance(p2, p0, p3)
    return max(dist_p1, dist_p2)


def register_bezier_curve(
    label: str,
    p1: Point,
    p2: Point,
    p3: Point,
    quality: str = "medium",
    force_recompute: bool = False,
    use_hybrid: bool = True,
    threshold: float = HYBRID_THRESHOLD,
    force_optimizer: Optional[str] = None,
):
    """
    Register a cubic Bezier curve for smooth movement patterns.
    Uses HYBRID selection to automatically choose between OLD and NEW optimizers.

    Args:
        label: Unique curve name (lowercase with underscores, e.g., "boss_charge")
        quality: Optimization quality ("fast", "medium", "high", "ultra")
        force_recompute: Force reoptimization even if curve already exists
        use_hybrid: Enable automatic optimizer selection (default: True)
        threshold: Distance threshold for optimizer selection (default: 0.5)
        force_optimizer: Force specific optimizer ("old" or "new"), overrides use_hybrid
    """
    p0: Point = (0, 0)
    
    if label in CURVE_PARAMS and not force_recompute:
        print(f"✓ Curve '{label}' already registered (use force_recompute=True to override)")
        return
    
    # Determine which optimizer to use
    if force_optimizer:
        use_new = (force_optimizer.lower() == "new")
        optimizer_name = force_optimizer.upper()
    elif use_hybrid:
        max_dist = calculate_curve_distance(p0, p1, p2, p3)
        use_new = max_dist < threshold
        optimizer_name = "NEW (basis-fit)" if use_new else "OLD (2-segment)"
        
        print(f"\n{'='*80}")
        print("HYBRID OPTIMIZER SELECTION")
        print(f"{'='*80}")
        print(f"Curve: {label}")
        print(f"Control points: p0={p0}, p1={p1}, p2={p2}, p3={p3}")
        print(f"Max distance from diagonal: {max_dist:.3f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"Selected optimizer: {optimizer_name}")
        print("=" * 80)
    else:
        use_new = False
        optimizer_name = "OLD (2-segment)"
    
    print(f"\nOptimizing curve '{label}' (quality={quality})...")
    
    # Use NEW optimizer (basis-fit)
    if use_new:
        CURVE_PARAMS[label] = register_bezier_curve_new(p0, p1, p2, p3, quality)
        _rewrite_curve_registry()
        
        print(f"\n✓ Curve '{label}' registered with NEW optimizer")
        print(f"⚠️  Restart your program to use CurveType.{label.upper()} enum")
        return
    
    # Use OLD optimizer (2-segment piecewise)

    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"quality must be one of {list(QUALITY_PRESETS.keys())}, got '{quality}'"
        )

    print("  OLD optimizer: 2-segment piecewise")

    def bezier_func(t: np.ndarray) -> np.ndarray:
        """Cubic Bezier: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃"""
        s = 1.0 - t
        return s**3 * p0[1] + 3 * s**2 * t * p1[1] + 3 * s * t**2 * p2[1] + t**3 * p3[1]

    settings = QUALITY_PRESETS[quality]
    best = optimize_curve_parameters(
        bezier_func,
        x0=0.0,
        x1=1.0,
        total_duration=1.0,
        y_exponents=settings["y_exps"],
        iterations=settings["iters"],
    )

    new_params: OldCurveParams = {
        "new_system": False,
        "t_split": float(best.t_split),
        "x_split": float(best.x_split),
        "r_in": float(best.r_in),
        "r_out": float(best.r_out),
        "y_coeffs_1": [float(c) for c in best.y_coeffs_1]
        if best.y_coeffs_1 is not None
        else [],
        "y_coeffs_2": [float(c) for c in best.y_coeffs_2]
        if best.y_coeffs_2 is not None
        else [],
        "y_exps": settings["y_exps"],
        "bezier_points": {"p0": list(p0), "p1": list(p1), "p2": list(p2), "p3": list(p3)},
        "quality_info": {
            "max_speed_dev": float(best.max_speed_dev),
            "rms_speed_dev": float(best.rms_speed_dev),
            "shape_rmse": 0.0,
            "score": float(best.score),
            "quality_preset": quality,
            "trigger_count": settings["triggers"],
        },
    }

    CURVE_PARAMS[label] = new_params  # type: ignore[typeddict-item]
    _rewrite_curve_registry()  # type: ignore[misc]

    print(f"\n✓ Curve '{label}' registered successfully!")
    print(f"  Max speed deviation: {best.max_speed_dev*100:.2f}%")
    print(f"  RMS speed deviation: {best.rms_speed_dev*100:.2f}%")
    print(f"  Trigger count: {settings['triggers']}")
    print(f"\n⚠️  Restart your program to use CurveType.{label.upper()} enum")
    print(f"   (or use string immediately: '{label}')")


def _rewrite_curve_registry() -> None:  # type: ignore[misc]
    """Rewrite auto-generated section to keep CurveType enum in sync w/ self-writing python."""
    my_path = Path(__file__)

    with open(my_path, "r") as f:
        lines = f.readlines()

    start_marker = "# AUTO-GENERATED SECTION - DO NOT EDIT BELOW THIS LINE\n"
    end_marker = "# END AUTO-GENERATED SECTION\n"

    try:
        start_idx = lines.index(start_marker)
        end_idx = lines.index(end_marker)
    except ValueError:
        raise RuntimeError(
            "Auto-generated section markers not found in movements.py!\n"
            "The file may have been manually edited."
        )

    if CURVE_PARAMS:
        enum_lines = ["class CurveType(Enum):\n"]
        for label in sorted(CURVE_PARAMS.keys()):
            enum_name = label.upper()
            enum_lines.append(f'    {enum_name} = "{label}"\n')
    else:
        enum_lines = ["class CurveType(Enum):\n", "    pass\n"]

    params_lines = ["CURVE_PARAMS = {\n"]
    for label in sorted(CURVE_PARAMS.keys()):
        p = CURVE_PARAMS[label]
        params_lines.append(f'    "{label}": {{\n')
        
        # Handle both OLD and NEW optimizer formats
        params_lines.append(f'        "new_system": {p["new_system"]},\n')
        
        if p["new_system"]:
            # NEW format: x_terms, y_terms
            assert p["new_system"] is True  # Type narrowing
            params_lines.append(f'        "x_terms": {p["x_terms"]},\n')
            params_lines.append(f'        "y_terms": {p["y_terms"]},\n')
        else:
            # OLD format: t_split, x_split, etc.
            assert p["new_system"] is False  # Type narrowing
            params_lines.append(
                f'        "t_split": {p["t_split"]}, "x_split": {p["x_split"]}, '
                f'"r_in": {p["r_in"]}, "r_out": {p["r_out"]},\n'
            )
            params_lines.append(f'        "y_coeffs_1": {p["y_coeffs_1"]},\n')
            params_lines.append(f'        "y_coeffs_2": {p["y_coeffs_2"]},\n')
            params_lines.append(f'        "y_exps": {p["y_exps"]},\n')
        
        params_lines.append(f'        "bezier_points": {p["bezier_points"]},\n')
        params_lines.append(f'        "quality_info": {p["quality_info"]}\n')
        params_lines.append("    },\n")
    params_lines.append("}\n")

    new_lines = (
        lines[: start_idx + 1]
        + ["\n"]
        + enum_lines
        + ["\n"]
        + params_lines
        + ["\n"]
        + lines[end_idx:]
    )

    with open(my_path, "w") as f:
        f.writelines(new_lines)


# ============================================================================
# CURVE APPLICATION (Runtime)
# ============================================================================

def _apply_basis_terms(
    component: Component,
    time: float,
    terms: List[BasisTermDict],
    dx: float,
    dy: float,
    duration: float,
) -> None:
    """Apply basis function terms as MoveBy triggers."""
    for term in terms:
        coef = float(term["coef"])
        easing, rate = term_to_gd_easing(term["kind"], float(term["rate"]))
        component.MoveBy(time, dx=dx * coef, dy=dy * coef, t=duration, type=easing, rate=rate)


def _apply_polynomial_terms(
    component: Component,
    time: float,
    coeffs: List[float],
    exponents: List[float],
    dy: float,
    duration: float,
) -> None:
    """Apply polynomial Y motion: y(t) = sum(coef_i * t^exp_i)."""
    for coef, exp in zip(coeffs, exponents):
        dy_segment = dy * float(coef)
        
        if abs(float(exp) - 1.0) < 1e-9:
            easing = e.Easing.NONE  # Linear term: t^1
            rate = 1.0
        else:
            easing = e.Easing.EASE_IN  # Polynomial term: t^exp
            rate = float(exp)
        
        component.MoveBy(time, dx=0, dy=dy_segment, t=duration, type=easing, rate=rate)


def _apply_new_optimizer(
    component: Component,
    time: float,
    params: NewCurveParams,
    dx: float,
    dy: float,
    duration: float,
) -> None:
    """Apply NEW optimizer (basis-fit) movement."""
    _apply_basis_terms(component, time, params["x_terms"], dx, 0.0, duration)
    _apply_basis_terms(component, time, params["y_terms"], 0.0, dy, duration)


def _apply_old_optimizer(
    component: Component,
    time: float,
    params: OldCurveParams,
    dx: float,
    dy: float,
    duration: float,
) -> None:
    """Apply OLD optimizer (2-segment piecewise) movement."""
    # Extract parameters
    T1 = duration * float(params["t_split"])
    T2 = duration * (1 - float(params["t_split"]))
    dx1 = dx * float(params["x_split"])
    dx2 = dx * (1 - float(params["x_split"]))
    
    # Segment 1: Ease-in X motion + polynomial Y motion
    component.MoveBy(time, dx=dx1, dy=0, t=T1, type=e.Easing.EASE_IN, rate=float(params["r_in"]))
    _apply_polynomial_terms(component, time, params["y_coeffs_1"], params["y_exps"], dy, T1)
    
    # Segment 2: Ease-out X motion + polynomial Y motion
    component.MoveBy(time, dx=dx2, dy=0, t=T2, type=e.Easing.EASE_OUT, rate=float(params["r_out"]))
    _apply_polynomial_terms(component, time, params["y_coeffs_2"], params["y_exps"], dy, T2)


def apply_bezier_movement(
    component: Component,
    time: float,
    curve_label: Union[CurveType, str],
    dx: float,
    dy: float,
    duration: float,
    generate_preview: bool = False,
) -> Component:
    """
    Use component.timed.BezierMove() instead of calling directly.
    Movement is RELATIVE displacement from current position.
    
    Automatically uses the correct optimizer (OLD or NEW) based on how
    the curve was registered.
    """
    if isinstance(curve_label, Enum):
        label = curve_label.value
    else:
        label = curve_label

    if label not in CURVE_PARAMS:
        available = list(CURVE_PARAMS.keys())
        raise KeyError(
            f"Curve '{label}' not registered!\n"
            f"Available curves: {available}\n"
            f"Register it with: register_bezier_curve('{label}', ...)\n"
            f"Or run: python setup_curves.py"
        )

    if component.target == -1:
        raise ValueError(
            "No target set! Use component.set_context(target=...) or "
            "component.temp_context() first"
        )

    params = CURVE_PARAMS[label]
    
    if params["new_system"]:
        assert params["new_system"] is True
        _apply_new_optimizer(component, time, params, dx, dy, duration)
    else:
        assert params["new_system"] is False
        _apply_old_optimizer(component, time, params, dx, dy, duration)

    if generate_preview:
        # Queue preview for batch parallel generation
        _preview_queue.append((label, params, dx, dy, duration))

    return component


# ============================================================================
# PREVIEW GENERATION (Optional - requires matplotlib)
# ============================================================================


def generate_previews_parallel(
    preview_specs: List[Tuple[str, CurveParams, float, float, float]],
    num_workers: Optional[int] = None,
):
    """Generate multiple curve previews in parallel."""
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Generating {len(preview_specs)} previews using {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(_generate_preview_worker, preview_specs)

    for _name, _success, message in results:
        print(message)

    successes = sum(1 for _, success, _ in results if success)
    print(f"\n✓ {successes}/{len(preview_specs)} previews generated successfully!")


def _generate_preview_worker(
    args: Tuple[str, CurveParams, float, float, float]
) -> Tuple[str, bool, str]:
    """Worker function for parallel preview generation."""
    curve_name, params, dx, dy, duration = args
    try:
        _generate_movement_preview(curve_name, params, dx, dy, duration)
        return (curve_name, True, f"✓ Preview saved for {curve_name}")
    except Exception as e:
        return (curve_name, False, f"✗ Failed for {curve_name}: {e}")


def _generate_movement_preview(
    curve_name: str, params: CurveParams, dx: float, dy: float, duration: float
):
    """Generate animated GIF preview. Saves to previews/{curve_name}_dx{dx}_dy{dy}.gif"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as e:
        print("⚠️  Preview generation requires matplotlib.")
        print("    Install with: pip install matplotlib")
        print(f"    Error: {e}")
        return

    print(f"Generating preview for '{curve_name}' movement...")
    
    # Generate actual bezier curve for comparison
    bp = params["bezier_points"]
    p0 = (bp["p0"][0], bp["p0"][1])
    p1 = (bp["p1"][0], bp["p1"][1])
    p2 = (bp["p2"][0], bp["p2"][1])
    p3 = (bp["p3"][0], bp["p3"][1])
    
    t_bezier = np.linspace(0, 1, 200)
    bezier_x, bezier_y = cubic_bezier_xy(t_bezier, p0, p1, p2, p3)
    bezier_path = np.column_stack([bezier_x * dx, bezier_y * dy])
    
    if params["new_system"]:
        # NEW optimizer: use basis terms
        assert params["new_system"] is True  # Type narrowing
        t_samples = np.linspace(0, 1, PREVIEW_SAMPLES)
        xi, yi = build_implemented_xy(t_samples, params["x_terms"], params["y_terms"])
        path = np.column_stack([xi * dx, yi * dy])
    else:
        # OLD optimizer: use 2-segment approach
        assert params["new_system"] is False  # Type narrowing
        T1 = duration * float(params["t_split"])
        T2 = duration * (1 - float(params["t_split"]))
        dx1 = dx * float(params["x_split"])
        dx2 = dx * (1 - float(params["x_split"]))

        exps = np.array(params["y_exps"], dtype=float)
        c1 = np.array(params["y_coeffs_1"], dtype=float)
        c2 = np.array(params["y_coeffs_2"], dtype=float)

        def position_at_time(t: float) -> Tuple[float, float]:
            if t <= T1:
                tau = t / T1 if T1 > 0 else 0
                x = dx1 * ease_in(np.array(tau), float(params["r_in"])).item()
                y = np.sum(c1 * (tau**exps)).item() * dy
            else:
                tau = (t - T1) / T2 if T2 > 0 else 0
                x = dx1 + dx2 * ease_out(np.array(tau), float(params["r_out"])).item()
                y = (np.sum(c1 * np.ones_like(exps)) + np.sum(c2 * (tau**exps))).item() * dy
            return x, y

        times = np.linspace(0, duration, PREVIEW_SAMPLES)
        path = np.array([position_at_time(t) for t in times])

    x_min, x_max = path[:, 0].min(), path[:, 0].max()
    y_min, y_max = path[:, 1].min(), path[:, 1].max()

    x_min = min(x_min, 0, dx)
    x_max = max(x_max, 0, dx)
    y_min = min(y_min, 0, dy)
    y_max = max(y_max, 0, dy)

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = x_range * PREVIEW_PADDING if x_range > 0 else 10
    y_pad = y_range * PREVIEW_PADDING if y_range > 0 else 10

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot actual bezier curve for comparison
    ax.plot(bezier_path[:, 0], bezier_path[:, 1], "g-", alpha=0.5, linewidth=2, label="Bezier curve (target)")
    
    # Plot implemented path
    ax.plot(path[:, 0], path[:, 1], "b-", alpha=0.3, linewidth=2, label="Implemented path")
    ax.plot([0], [0], "go", markersize=10, label="Start (0,0)")
    ax.plot([dx], [dy], "ro", markersize=10, label=f"End ({dx:.0f},{dy:.0f})")

    (current_point,) = ax.plot([], [], "bo", markersize=8)
    (trail,) = ax.plot([], [], "b-", alpha=0.6, linewidth=1)

    bezier_pts = params.get("bezier_points", {})
    if bezier_pts:
        p0 = bezier_pts["p0"]
        p1 = bezier_pts["p1"]
        p2 = bezier_pts["p2"]
        p3 = bezier_pts["p3"]

        ctrl_x = [p0[0] * dx, p1[0] * dx, p2[0] * dx, p3[0] * dx]
        ctrl_y = [p0[1] * dy, p1[1] * dy, p2[1] * dy, p3[1] * dy]
        ax.plot(ctrl_x, ctrl_y, "r--", alpha=0.3, linewidth=1, label="Control points")
        ax.plot(ctrl_x, ctrl_y, "rs", alpha=0.5, markersize=6)

    ax.set_xlabel("X Position (relative)")
    ax.set_ylabel("Y Position (relative)")
    ax.set_title(
        f"Bezier Movement: {curve_name}\ndx={dx}, dy={dy}, duration={duration}s"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def init():
        current_point.set_data([], [])
        trail.set_data([], [])
        return current_point, trail

    def update(frame: int) -> Tuple[Any, Any]:
        idx = int((frame / (PREVIEW_FRAMES - 1)) * (PREVIEW_SAMPLES - 1))
        current_point.set_data([path[idx, 0]], [path[idx, 1]])
        trail.set_data(path[: idx + 1, 0], path[: idx + 1, 1])
        return current_point, trail

    try:
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=PREVIEW_FRAMES,
            interval=50,
            blit=True,
        )

        output_dir = Path("previews")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{curve_name}_dx{int(dx)}_dy{int(dy)}.gif"

        writer = PillowWriter(fps=PREVIEW_FPS)
        anim.save(output_path, writer=writer)

        print(f"✓ Preview saved to: {output_path}")
    finally:
        plt.close(fig)
        plt.close("all")


def list_curves() -> None:
    """Print all registered curves."""
    if not CURVE_PARAMS:
        print("No curves registered yet.")
        print("Use register_bezier_curve() to add curves or run: python setup_curves.py")
        return

    print(f"\nRegistered Curves ({len(CURVE_PARAMS)} total):")
    print("=" * 80)

    for label in sorted(CURVE_PARAMS.keys()):
        params = CURVE_PARAMS[label]
        quality_info = params["quality_info"]
        bp = params["bezier_points"]

        print(f"\n{label.upper()}")
        print(f"  Enum: CurveType.{label.upper()}")
        print(f"  String: '{label}'")
        print(f"  Triggers: {quality_info['trigger_count']}")
        print(f"  Quality: {quality_info['quality_preset']}")
        print(f"  Max speed dev: {quality_info['max_speed_dev']*100:.2f}%")
        print(
            f"  Bezier: p0={bp['p0']}, p1={bp['p1']}, "
            f"p2={bp['p2']}, p3={bp['p3']}"
        )

    print("\n" + "=" * 80)


def wait_for_previews(num_workers: Optional[int] = None) -> None:
    """Generate all queued previews in parallel using multiprocessing."""
    if not _preview_queue:
        return
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Generating {len(_preview_queue)} previews in parallel using {num_workers} workers...")
    
    # Use 'fork' context to avoid pickling issues
    ctx = get_context('fork')
    with ctx.Pool(num_workers) as pool:
        results = pool.map(_generate_preview_worker, _preview_queue)
    
    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes
    
    if failures > 0:
        print(f"✓ {successes}/{len(_preview_queue)} previews generated successfully ({failures} failed)")
    else:
        print(f"✓ All {successes} previews generated successfully")
    
    _preview_queue.clear()
