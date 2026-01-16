#!/usr/bin/env python3
"""
Setup script to register common Bezier curves for the movement system.

Run this once to pre-compute optimal parameters for all curves.
After running, restart your program to use the CurveType enum.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import touhou_scs
sys.path.insert(0, str(Path(__file__).parent.parent))

from touhou_scs.movements import register_bezier_curve, list_curves


def register_common_curves():
    """Register commonly used movement curves."""
    
    print("="*80)
    print("REGISTERING COMMON BEZIER CURVES")
    print("="*80)
    
    # Gentle arcs (for fairy entrances)
    register_bezier_curve("gentle_arc",
        p1=(0.2, 0.6),   # Pull up early
        p2=(0.8, 1.2),   # Pull up late
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    register_bezier_curve("gentle_arc_down",
        p1=(0.2, -0.3),  # Pull down early
        p2=(0.8, 0.7),   # Pull up to end
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    # Steep movements (for fast enemies)
    register_bezier_curve("steep_dive",
        p1=(0.3, -0.5),  # Sharp dive
        p2=(0.7, 0.5),   # Sharp recover
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    register_bezier_curve("steep_rise",
        p1=(0.3, 1.5),   # Sharp up
        p2=(0.7, 0.5),   # Ease down to end
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    # S-curves (for weaving patterns)
    register_bezier_curve("s_curve",
        p1=(0.2, 1.5),   # Overshoot up
        p2=(0.8, -0.5),  # Overshoot down
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    register_bezier_curve("s_curve_reverse",
        p1=(0.2, -0.5),  # Overshoot down
        p2=(0.8, 1.5),   # Overshoot up
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    # Smooth easing (for natural movement)
    register_bezier_curve("smooth_ease",
        p1=(0.33, 0.0),  # Flat start
        p2=(0.67, 1.0),  # Flat end
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    # Boss patterns (high quality)
    register_bezier_curve("boss_weave",
        p1=(0.25, 1.2),  # Up
        p2=(0.75, -0.2), # Down
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    register_bezier_curve("boss_charge",
        p1=(0.1, 0.1),   # Slow start
        p2=(0.5, 0.8),   # Accelerate
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    # Fast/low-precision variants (for background enemies)
    register_bezier_curve("fast_arc",
        p1=(0.2, 0.6),
        p2=(0.8, 1.2),
        p3=(1, 1),
        quality="ultra",
        force_recompute=True
    )
    
    print("\n" + "="*80)
    print("REGISTRATION COMPLETE!")


if __name__ == "__main__":
    import time
    start_time = time.time()
    register_common_curves()
    print()
    list_curves()
    end_time = time.time()
    print(f"\nTotal registration time: {end_time - start_time:.2f} seconds")
