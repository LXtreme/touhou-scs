# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Bezier curve movement system with auto-generated curve parameters (v2).

Upgrades vs previous version:
- Correct Bezier math: optimizes against the true 2D cubic Bezier using arc-length parameter τ.
- Stored params now genuinely represent the curve's geometry.
- Previews show: true Bezier (scaled) + implemented path (scaled).
"""

from __future__ import annotations
from typing import Union, TYPE_CHECKING, TypedDict, List, Any
from enum import Enum
from pathlib import Path
import hashlib
import json
import numpy as np

from touhou_scs.curve_optimizer_v2 import (
    resample_bezier_by_arclength,
    optimize_bezier_profile,
    implemented_path_samples,
    PieceParams,
)

if TYPE_CHECKING:
    from touhou_scs.component import Component


class BezierPoints(TypedDict):
    p0: List[float]
    p1: List[float]
    p2: List[float]
    p3: List[float]


class QualityInfo(TypedDict):
    max_speed_dev: float
    rms_speed_dev: float
    shape_rmse: float
    score: float
    quality_preset: str
    trigger_count: int
    optimizer_version: str
    curve_hash: str


class CurveParams(TypedDict):
    t_split: float
    x_split: float
    r_in: float
    r_out: float
    y_coeffs_1: List[float]
    y_coeffs_2: List[float]
    y_exps: List[float]
    bezier_points: BezierPoints
    quality_info: QualityInfo


# =========================================================================
# AUTO-GENERATED SECTION - DO NOT EDIT BELOW THIS LINE

class CurveType(Enum):
    # Populated by _rewrite_self()
    pass


CURVE_PARAMS: dict[str, Any] = {}  # type: ignore

# END AUTO-GENERATED SECTION
# =========================================================================

CURVE_PARAMS: dict[str, Any] = CURVE_PARAMS  # type: ignore

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    # Total triggers = 2 (X) + 2*len(y_exps) (Y)
    "fast":   {"iters": 1200, "y_exps": [1.0, 2.0],              "triggers": 6},
    "medium": {"iters": 2000, "y_exps": [1.0, 2.0, 3.0],         "triggers": 8},
    "high":   {"iters": 6000, "y_exps": [1.0, 2.0, 3.0],         "triggers": 8},
    "ultra":  {"iters": 9000, "y_exps": [1.0, 2.0, 3.0, 4.0],    "triggers": 10},
}

Point = tuple[float, float]
_OPTIMIZER_VERSION = "bezier_xy_arclength_v1"


def _hash_curve(p0: Point, p1: Point, p2: Point, p3: Point, quality: str) -> str:
    payload = {"p0": p0, "p1": p1, "p2": p2, "p3": p3, "quality": quality, "v": _OPTIMIZER_VERSION}
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def register_bezier_curve(
    label: str,
    p0: Point, p1: Point, p2: Point, p3: Point,
    quality: str = "medium",
    force_recompute: bool = False
) -> None:
    """Register and optimize a cubic Bezier curve. Restart program to use CurveType enum."""
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"quality must be one of {list(QUALITY_PRESETS.keys())}, got '{quality}'")

    curve_hash = _hash_curve(p0, p1, p2, p3, quality)

    if label in CURVE_PARAMS and not force_recompute:
        old = CURVE_PARAMS[label]
        old_hash = old.get("quality_info", {}).get("curve_hash", "")
        if old_hash == curve_hash:
            print(f"✓ Curve '{label}' already registered (hash match).")
            return
        print(f"⚠️  Curve '{label}' exists but hash differs; recompute or force_recompute=True.")
        return

    print(f"Optimizing curve '{label}' (quality={quality})...")
    print(f"  Bezier control points: p0={p0}, p1={p1}, p2={p2}, p3={p3}")

    tau, x_tau, y_tau = resample_bezier_by_arclength(p0, p1, p2, p3, dense=5000, out=650)

    settings = QUALITY_PRESETS[quality]
    best = optimize_bezier_profile(
        tau, x_tau, y_tau,
        T=1.0,
        y_exps=settings["y_exps"],
        iters=settings["iters"],
        seed=1,
        t_split_range=(0.3, 0.7),
        x_split_range=(0.2, 0.8),
        rate_range=(0.5, 10.0),
        verbose=True,
    )

    new_params: Any = {
        "t_split": float(best.t_split),
        "x_split": float(best.x_split),
        "r_in": float(best.r_in),
        "r_out": float(best.r_out),
        "y_coeffs_1": [float(c) for c in (best.y_coeffs_1.tolist() if best.y_coeffs_1 is not None else [])],
        "y_coeffs_2": [float(c) for c in (best.y_coeffs_2.tolist() if best.y_coeffs_2 is not None else [])],
        "y_exps": settings["y_exps"],
        "bezier_points": {"p0": list(p0), "p1": list(p1), "p2": list(p2), "p3": list(p3)},
        "quality_info": {
            "max_speed_dev": float(best.max_dev),
            "rms_speed_dev": float(best.rms_dev),
            "shape_rmse": float(best.shape_rmse),
            "score": float(best.score),
            "quality_preset": quality,
            "trigger_count": settings["triggers"],
            "optimizer_version": _OPTIMIZER_VERSION,
            "curve_hash": curve_hash,
        }
    }

    CURVE_PARAMS[label] = new_params  # type: ignore
    _rewrite_self()

    print(f"✓ Curve '{label}' registered!")
    qi = new_params["quality_info"]
    print(f"  Max speed deviation: {qi['max_speed_dev']*100:.2f}%")
    print(f"  RMS speed deviation: {qi['rms_speed_dev']*100:.2f}%")
    print(f"  Shape RMSE: {qi['shape_rmse']:.5f} (normalized)")
    print(f"  Trigger count: {qi['trigger_count']}")
    print(f"⚠️  Restart your program to use CurveType.{label.upper()} enum")
    print(f"   (or use string immediately: '{label}')")


def apply_bezier_movement(
    component: Component,
    time: float,
    curve_label: Union["CurveType", str],
    dx: float,
    dy: float,
    duration: float,
    generate_preview: bool = False
) -> Component:
    """Apply Bezier curve movement (relative displacement)."""
    from touhou_scs import enums as e

    label = curve_label.value if isinstance(curve_label, Enum) else curve_label
    if label not in CURVE_PARAMS:
        available = list(CURVE_PARAMS.keys())
        raise KeyError(
            f"Curve '{label}' not registered!\n"
            f"Available curves: {available}\n"
            f"Register it with: register_bezier_curve('{label}', ...)"
        )

    params = CURVE_PARAMS[label]
    if component.target == -1:
        raise ValueError("No target set! Use component.set_context(target=...) or temp_context() first")

    t_split = float(params["t_split"])
    x_split = float(params["x_split"])
    r_in = float(params["r_in"])
    r_out = float(params["r_out"])

    y_exps = [float(x) for x in params["y_exps"]]
    c1 = [float(x) for x in params["y_coeffs_1"]]
    c2 = [float(x) for x in params["y_coeffs_2"]]

    T1 = duration * t_split
    T2 = duration * (1.0 - t_split)

    dx1 = dx * x_split
    dx2 = dx * (1.0 - x_split)

    component.MoveBy(time, dx=dx1, dy=0.0, t=T1, type=e.Easing.EASE_IN, rate=r_in)
    component.MoveBy(time + T1, dx=dx2, dy=0.0, t=T2, type=e.Easing.EASE_OUT, rate=r_out)

    for coef, exp in zip(c1, y_exps):
        dy_seg = dy * coef
        if abs(exp - 1.0) < 1e-9:
            component.MoveBy(time, dx=0.0, dy=dy_seg, t=T1, type=e.Easing.NONE, rate=1.0)
        else:
            component.MoveBy(time, dx=0.0, dy=dy_seg, t=T1, type=e.Easing.EASE_IN, rate=exp)

    for coef, exp in zip(c2, y_exps):
        dy_seg = dy * coef
        if abs(exp - 1.0) < 1e-9:
            component.MoveBy(time + T1, dx=0.0, dy=dy_seg, t=T2, type=e.Easing.NONE, rate=1.0)
        else:
            component.MoveBy(time + T1, dx=0.0, dy=dy_seg, t=T2, type=e.Easing.EASE_IN, rate=exp)

    if generate_preview:
        _generate_movement_preview(label, params, dx, dy, duration)

    return component


def _generate_movement_preview(curve_name: str, params: Any, dx: float, dy: float, duration: float) -> None:
    """Generate animated GIF preview comparing true Bezier vs implemented path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as exc:
        print("⚠️  Preview generation requires matplotlib. Install with: pip install matplotlib")
        print(f"    Error: {exc}")
        return

    bp = params.get("bezier_points", {})
    if not bp:
        print("⚠️  No bezier points stored; cannot preview.")
        return

    p0 = tuple(bp["p0"]); p1 = tuple(bp["p1"]); p2 = tuple(bp["p2"]); p3 = tuple(bp["p3"])
    tau_t, x_t, y_t = resample_bezier_by_arclength(p0, p1, p2, p3, dense=6000, out=500)

    pp = PieceParams(
        t_split=float(params["t_split"]),
        x_split=float(params["x_split"]),
        r_in=float(params["r_in"]),
        r_out=float(params["r_out"]),
    )
    pp.y_coeffs_1 = np.array(params["y_coeffs_1"], dtype=float)
    pp.y_coeffs_2 = np.array(params["y_coeffs_2"], dtype=float)
    y_exps = [float(x) for x in params["y_exps"]]
    _, x_i, y_i = implemented_path_samples(pp, y_exps, n=500, T=1.0)

    true_xy = np.column_stack([x_t * dx, y_t * dy])
    impl_xy = np.column_stack([x_i * dx, y_i * dy])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(true_xy[:, 0], true_xy[:, 1], "g-", linewidth=2, label="True Bezier (arc-length)")
    ax.plot(impl_xy[:, 0], impl_xy[:, 1], "b--", linewidth=2, alpha=0.85, label="Implemented (triggers)")

    ctrl_x = [p0[0]*dx, p1[0]*dx, p2[0]*dx, p3[0]*dx]
    ctrl_y = [p0[1]*dy, p1[1]*dy, p2[1]*dy, p3[1]*dy]
    ax.plot(ctrl_x, ctrl_y, "r--", alpha=0.35, linewidth=1, label="Bezier control")
    ax.plot(ctrl_x, ctrl_y, "rs", alpha=0.5, markersize=6)

    pt_true, = ax.plot([], [], "go", markersize=7)
    pt_impl, = ax.plot([], [], "bo", markersize=7)
    trail_true, = ax.plot([], [], "g-", alpha=0.25, linewidth=1)
    trail_impl, = ax.plot([], [], "b-", alpha=0.25, linewidth=1)

    qi = params.get("quality_info", {})
    ax.set_title(
        f"Bezier Movement: {curve_name}\n"
        f"dx={dx}, dy={dy}, duration={duration}s | "
        f"max_dev={qi.get('max_speed_dev',0)*100:.1f}% shape_rmse={qi.get('shape_rmse',0):.4f}"
    )
    ax.set_xlabel("X (relative)")
    ax.set_ylabel("Y (relative)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    allx = np.concatenate([true_xy[:, 0], impl_xy[:, 0], np.array(ctrl_x)])
    ally = np.concatenate([true_xy[:, 1], impl_xy[:, 1], np.array(ctrl_y)])
    x_min, x_max = float(allx.min()), float(allx.max())
    y_min, y_max = float(ally.min()), float(ally.max())
    pad_x = 0.15 * (x_max - x_min + 1e-9)
    pad_y = 0.15 * (y_max - y_min + 1e-9)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    frames = 120
    def init():
        pt_true.set_data([], []); pt_impl.set_data([], [])
        trail_true.set_data([], []); trail_impl.set_data([], [])
        return pt_true, pt_impl, trail_true, trail_impl

    def update(frame: int):
        idx = int((frame / (frames - 1)) * (len(true_xy) - 1))
        pt_true.set_data([true_xy[idx, 0]], [true_xy[idx, 1]])
        pt_impl.set_data([impl_xy[idx, 0]], [impl_xy[idx, 1]])
        trail_true.set_data(true_xy[:idx+1, 0], true_xy[:idx+1, 1])
        trail_impl.set_data(impl_xy[:idx+1, 0], impl_xy[:idx+1, 1])
        return pt_true, pt_impl, trail_true, trail_impl

    try:
        anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=33, blit=True)
        output_dir = Path("previews"); output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{curve_name}_dx{int(dx)}_dy{int(dy)}.gif"
        anim.save(output_path, writer=PillowWriter(fps=30))
        print(f"✓ Preview saved to: {output_path}")
    except Exception as exc:
        print(f"⚠️  Failed to generate preview: {exc}")
    finally:
        plt.close(fig); plt.close("all")


def _rewrite_self() -> None:
    """Rewrite the auto-generated section."""
    my_path = Path(__file__)
    with open(my_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_marker = "# AUTO-GENERATED SECTION - DO NOT EDIT BELOW THIS LINE\n"
    end_marker = "# END AUTO-GENERATED SECTION\n"
    try:
        start_idx = lines.index(start_marker)
        end_idx = lines.index(end_marker)
    except ValueError:
        raise RuntimeError("Auto-generated section markers not found. The file may have been edited.")

    if CURVE_PARAMS:
        enum_lines = ["class CurveType(Enum):\n"]
        for label in sorted(CURVE_PARAMS.keys()):
            enum_lines.append(f'    {label.upper()} = "{label}"\n')
    else:
        enum_lines = ["class CurveType(Enum):\n", "    pass\n"]

    params_lines = ["CURVE_PARAMS = {\n"]
    for label in sorted(CURVE_PARAMS.keys()):
        p = CURVE_PARAMS[label]
        params_lines.append(f'    "{label}": {{\n')
        params_lines.append(f'        "t_split": {p["t_split"]}, "x_split": {p["x_split"]}, "r_in": {p["r_in"]}, "r_out": {p["r_out"]},\n')
        params_lines.append(f'        "y_coeffs_1": {p["y_coeffs_1"]},\n')
        params_lines.append(f'        "y_coeffs_2": {p["y_coeffs_2"]},\n')
        params_lines.append(f'        "y_exps": {p["y_exps"]},\n')
        params_lines.append(f'        "bezier_points": {p["bezier_points"]},\n')
        params_lines.append(f'        "quality_info": {p["quality_info"]},\n')
        params_lines.append("    },\n")
    params_lines.append("}\n")

    new_lines = lines[:start_idx + 1] + ["\n"] + enum_lines + ["\n"] + params_lines + ["\n"] + lines[end_idx:]
    with open(my_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def list_curves() -> None:
    """Print all registered curves with their quality info."""
    if not CURVE_PARAMS:
        print("No curves registered yet.")
        return
    print(f"\nRegistered Curves ({len(CURVE_PARAMS)} total):")
    print("=" * 80)
    for label in sorted(CURVE_PARAMS.keys()):
        params = CURVE_PARAMS[label]
        qi = params.get("quality_info", {})
        print(f"\n{label.upper()}  triggers={qi.get('trigger_count','?')}  "
              f"max_dev={qi.get('max_speed_dev',0)*100:.2f}%  shape={qi.get('shape_rmse',0):.5f}")
    print("\n" + "=" * 80)


__all__ = [
    "CurveType",
    "QUALITY_PRESETS",
    "register_bezier_curve",
    "apply_bezier_movement",
    "list_curves",
]
