# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""
Bezier curve movement system for smooth, speed-consistent object motion.

SYSTEM OVERVIEW:
    1. Define cubic Bezier curves with 4 control points (p0, p1, p2, p3)
    2. Optimization splits curve into 2 piecewise segments with ease-in/ease-out
    3. Each segment is fit with polynomial approximations for Y motion
    4. Parameters are stored in CURVE_PARAMS dict and CurveType enum (auto-generated)
    5. At runtime, apply_bezier_movement() generates multiple MoveBy triggers
    6. Optional: Generate animated GIF previews for visual debugging

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
from typing import Union, TYPE_CHECKING, List, Any, Callable, Optional, Tuple, TypedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from touhou_scs import enums as e

if TYPE_CHECKING: from touhou_scs.component import Component


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
PREVIEW_FPS = 20
PREVIEW_FRAMES = 120
PREVIEW_SAMPLES = 100
PREVIEW_PADDING = 0.15

# Optimization search ranges
SPLIT_TIME_RANGE = (0.3, 0.7)  # Where to split the curve (30%-70% of duration)
SPLIT_DISTANCE_RANGE = (0.3, 0.7)  # Where to split X distance
EASING_RATE_RANGE = (0.5, 10.0)  # Ease-in/ease-out exponent range

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

Point = tuple[float | int, float | int]


class BezierPointsDict(TypedDict):
    p0: list[float]
    p1: list[float]
    p2: list[float]
    p3: list[float]


class QualityInfoDict(TypedDict):
    max_speed_dev: float
    rms_speed_dev: float
    score: float
    quality_preset: str
    trigger_count: int


class CurveParamsDict(TypedDict):
    t_split: float
    x_split: float
    r_in: float
    r_out: float
    y_coeffs_1: list[float]
    y_coeffs_2: list[float]
    y_exps: list[float]
    bezier_points: BezierPointsDict
    quality_info: QualityInfoDict


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


def ease_in(t: np.ndarray, rate: float) -> np.ndarray:
    return t**rate


def ease_out(t: np.ndarray, rate: float) -> np.ndarray:
    return 1.0 - (1.0 - t) ** rate


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
        "t_split": 0.3838947235978839, "x_split": 0.3804607334987112, "r_in": 0.9297341708340849, "r_out": 1.022084549655822,
        "y_coeffs_1": [0.1466912954202604, 0.22918905778951792, -0.06187573143066313],
        "y_coeffs_2": [0.7565234623935764, 0.1998441479120018, -0.2704433702594306],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.1, 0.1], 'p2': [0.5, 0.8], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.2464665260393245, 'rms_speed_dev': 0.11874152066456799, 'score': 0.611674572743217, 'quality_preset': 'high', 'trigger_count': 8}
    },
    "boss_weave": {
        "t_split": 0.5952580375632388, "x_split": 0.5919327032262545, "r_in": 1.4668953134522453, "r_out": 1.6632866820264753,
        "y_coeffs_1": [0.9186746398741304, 0.3934363038669184, -0.8762917678408557],
        "y_coeffs_2": [-0.2836421563933917, 1.7812626177790671, -0.9584556523493157],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.25, 1.2], 'p2': [0.75, -0.2], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.4105148078315769, 'rms_speed_dev': 0.16711825332015265, 'score': 0.9881478689833064, 'quality_preset': 'high', 'trigger_count': 8}
    },
    "fast_arc": {
        "t_split": 0.5528564350939283, "x_split": 0.5105509747297531, "r_in": 1.2457192251378935, "r_out": 1.1917087327317615,
        "y_coeffs_1": [0.6618016713482816, 0.17567463613154619],
        "y_coeffs_2": [0.6755812585833344, -0.4981575199606615],
        "y_exps": [1.0, 2.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, 0.6], 'p2': [0.8, 1.2], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.4707837301734764, 'rms_speed_dev': 0.24105342713343872, 'score': 1.1826208874803916, 'quality_preset': 'fast', 'trigger_count': 6}
    },
    "gentle_arc": {
        "t_split": 0.5528564350939283, "x_split": 0.5105509747297531, "r_in": 1.2457192251378935, "r_out": 1.1917087327317615,
        "y_coeffs_1": [0.49119675848491606, 0.7429430977710766, -0.42439216075615965],
        "y_coeffs_2": [0.7086284979781838, -0.6080409713707825, 0.08220741770220086],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, 0.6], 'p2': [0.8, 1.2], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.5364202642004807, 'rms_speed_dev': 0.24572297288869338, 'score': 1.3185635012896548, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "gentle_arc_down": {
        "t_split": 0.4650491903889169, "x_split": 0.5375265415605261, "r_in": 0.903234583964718, "r_out": 1.0903231509953943,
        "y_coeffs_1": [-0.5161921199223339, 1.3333059734239754, -0.4890148079219687],
        "y_coeffs_2": [0.7911445030008714, 0.10511012582646295, -0.2282173425915899],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, -0.3], 'p2': [0.8, 0.7], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.4776020580499446, 'rms_speed_dev': 0.16207447537003825, 'score': 1.1172785914699275, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "s_curve": {
        "t_split": 0.5, "x_split": 0.5, "r_in": 2.0, "r_out": 2.0,
        "y_coeffs_1": [-0.029299634442950036, 2.8672048633074065, -2.3629764576488546],
        "y_coeffs_2": [-1.188270279723602, 3.8155785782043488, -2.117274868510437],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, 1.5], 'p2': [0.8, -0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.9838946694419595, 'rms_speed_dev': 0.31205246393422753, 'score': 2.2798418028181464, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "s_curve_reverse": {
        "t_split": 0.4650491903889169, "x_split": 0.5375265415605261, "r_in": 0.903234583964718, "r_out": 1.0903231509953943,
        "y_coeffs_1": [-0.8325307478570174, 2.450585000924388, -1.0421124186552628],
        "y_coeffs_2": [1.1628343850133565, -0.3992341426047321, -0.3557240723146294],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.2, -0.5], 'p2': [0.8, 1.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.5214625650303719, 'rms_speed_dev': 0.32431766258433314, 'score': 1.367242792645077, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "smooth_ease": {
        "t_split": 0.4650491903889169, "x_split": 0.5375265415605261, "r_in": 0.903234583964718, "r_out": 1.0903231509953943,
        "y_coeffs_1": [0.06733997444941249, 0.8392950655070808, -0.3524985276630442],
        "y_coeffs_2": [0.7641889880122802, -0.15677360776983518, -0.1654510516411963],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.33, 0.0], 'p2': [0.67, 1.0], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.6318140878494426, 'rms_speed_dev': 0.2395457529712413, 'score': 1.5031739286701264, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "steep_dive": {
        "t_split": 0.5528564350939283, "x_split": 0.5105509747297531, "r_in": 1.2457192251378935, "r_out": 1.1917087327317615,
        "y_coeffs_1": [-0.5103624838867473, 0.33707657413929126, 0.3232276024042645],
        "y_coeffs_2": [0.9017470463221815, 0.33935388480842377, -0.3795555095208024],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.3, -0.5], 'p2': [0.7, 0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.47539309683639686, 'rms_speed_dev': 0.3609109570703456, 'score': 1.3116971507431394, 'quality_preset': 'medium', 'trigger_count': 8}
    },
    "steep_rise": {
        "t_split": 0.5528564350939283, "x_split": 0.5105509747297531, "r_in": 1.2457192251378935, "r_out": 1.1917087327317615,
        "y_coeffs_1": [1.4231522582118639, 0.0953692959804745, -0.6640255498056185],
        "y_coeffs_2": [-0.08941280599331476, -0.07512421777244091, 0.3024041282093851],
        "y_exps": [1.0, 2.0, 3.0],
        "bezier_points": {'p0': [0, 0], 'p1': [0.3, 1.5], 'p2': [0.7, 0.5], 'p3': [1, 1]},
        "quality_info": {'max_speed_dev': 0.6431730813518954, 'rms_speed_dev': 0.33929250244503223, 'score': 1.625638665148823, 'quality_preset': 'medium', 'trigger_count': 8}
    },
}

# END AUTO-GENERATED SECTION
# ============================================================================

# Type annotation for the auto-generated CURVE_PARAMS
CURVE_PARAMS: dict[str, CurveParamsDict]


# ============================================================================
# CURVE REGISTRATION
# ============================================================================


def register_bezier_curve(
    label: str,
    p1: Point,
    p2: Point,
    p3: Point,
    quality: str = "medium",
    force_recompute: bool = False,
):
    """
    Register a cubic Bezier curve for smooth movement patterns.
    Requires program restart to use the new CurveType enum value
    
    Args:
        label: Unique curve name (lowercase with underscores, e.g., "boss_charge")
        quality: Optimization quality ("fast", "medium", "high", "ultra")
                 Higher quality = more triggers, better speed consistency, slower optimization
    """
    p0: Point = (0, 0)

    if label in CURVE_PARAMS and not force_recompute:
        print(f"✓ Curve '{label}' already registered (use force_recompute=True to override)")
        return

    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"quality must be one of {list(QUALITY_PRESETS.keys())}, got '{quality}'"
        )

    print(f"\nOptimizing curve '{label}' (quality={quality})...")
    print(f"  Bezier control points: p0={p0}, p1={p1}, p2={p2}, p3={p3}")

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

    new_params: CurveParamsDict = {
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
            "score": float(best.score),
            "quality_preset": quality,
            "trigger_count": settings["triggers"],
        },
    }

    CURVE_PARAMS[label] = new_params  # type: ignore
    _rewrite_curve_registry()

    print(f"\n✓ Curve '{label}' registered successfully!")
    print(f"  Max speed deviation: {best.max_speed_dev*100:.2f}%")
    print(f"  RMS speed deviation: {best.rms_speed_dev*100:.2f}%")
    print(f"  Trigger count: {settings['triggers']}")
    print(f"\n⚠️  Restart your program to use CurveType.{label.upper()} enum")
    print(f"   (or use string immediately: '{label}')")


def _rewrite_curve_registry() -> None:
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

    params: CurveParamsDict = CURVE_PARAMS[label]
    target = component.target

    if target == -1:
        raise ValueError(
            "No target set! Use component.set_context(target=...) or "
            "component.temp_context() first"
        )

    # Extract parameters
    T1 = duration * float(params["t_split"])
    T2 = duration * (1 - float(params["t_split"]))
    dx1 = dx * float(params["x_split"])
    dx2 = dx * (1 - float(params["x_split"]))

    # Segment 1: Ease-in X motion + polynomial Y motion
    component.MoveBy(
        time, dx=dx1, dy=0, t=T1, type=e.Easing.EASE_IN, rate=float(params["r_in"])
    )

    # Apply polynomial Y motion: y(t) = sum(coef_i * t^exp_i)
    # Linear terms (exp=1.0) use NONE easing, higher powers use EASE_IN to match the exponent
    for coef, exp in zip(params["y_coeffs_1"], params["y_exps"]):
        dy_segment = dy * float(coef)

        if abs(float(exp) - 1.0) < 1e-9:
            easing = e.Easing.NONE  # Linear term: t^1
            rate = 1.0
        else:
            easing = e.Easing.EASE_IN  # Polynomial term: t^exp
            rate = float(exp)

        component.MoveBy(time, dx=0, dy=dy_segment, t=T1, type=easing, rate=rate)

    # Segment 2: Ease-out X motion + polynomial Y motion
    component.MoveBy(
        time, dx=dx2, dy=0, t=T2, type=e.Easing.EASE_OUT, rate=float(params["r_out"])
    )

    for coef, exp in zip(params["y_coeffs_2"], params["y_exps"]):
        dy_segment = dy * float(coef)

        if abs(float(exp) - 1.0) < 1e-9:
            easing = e.Easing.NONE  # Linear term: t^1
            rate = 1.0
        else:
            easing = e.Easing.EASE_IN  # Polynomial term: t^exp
            rate = float(exp)

        component.MoveBy(time, dx=0, dy=dy_segment, t=T2, type=easing, rate=rate)

    if generate_preview:
        _generate_movement_preview(label, params, dx, dy, duration)

    return component


# ============================================================================
# PREVIEW GENERATION (Optional - requires matplotlib)
# ============================================================================


def generate_previews_parallel(
    preview_specs: List[Tuple[str, CurveParamsDict, float, float, float]],
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
    args: Tuple[str, CurveParamsDict, float, float, float]
) -> Tuple[str, bool, str]:
    """Worker function for parallel preview generation."""
    curve_name, params, dx, dy, duration = args
    try:
        _generate_movement_preview(curve_name, params, dx, dy, duration)
        return (curve_name, True, f"✓ Preview saved for {curve_name}")
    except Exception as e:
        return (curve_name, False, f"✗ Failed for {curve_name}: {e}")


def _generate_movement_preview(
    curve_name: str, params: CurveParamsDict, dx: float, dy: float, duration: float
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

    ax.plot(path[:, 0], path[:, 1], "b-", alpha=0.3, linewidth=2, label="Path")
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
        ax.plot(ctrl_x, ctrl_y, "r--", alpha=0.3, linewidth=1, label="Bezier control")
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
        params: CurveParamsDict = CURVE_PARAMS[label]
        quality_info: QualityInfoDict = params["quality_info"]
        bp: BezierPointsDict = params["bezier_points"]

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
