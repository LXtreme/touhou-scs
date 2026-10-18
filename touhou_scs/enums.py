"""
Touhou SCS - Enums and Constants Module

All game constants, property IDs, and enumerated values.
Central location for magic numbers with type safety and IDE autocomplete.
"""

from typing import Final
from gmdbuilder import enum as gdenum

Item = gdenum.Item

POINTER_OBJ_ID = 3802

# ============================================================================
# EASING TYPES
# ============================================================================

class Easing:
    """Easing curve types for trigger animations"""
    NONE = 0
    EASE_IN_OUT = 1
    EASE_IN = 2
    EASE_OUT = 3
    ELASTIC_IN_OUT = 4
    ELASTIC_IN = 5
    ELASTIC_OUT = 6
    BOUNCE_IN_OUT = 7
    BOUNCE_IN = 8
    BOUNCE_OUT = 9
    EXPONENTIAL_IN_OUT = 10
    EXPONENTIAL_IN = 11
    EXPONENTIAL_OUT = 12
    SINE_IN_OUT = 13
    SINE_IN = 14
    SINE_OUT = 15
    BACK_IN_OUT = 16
    BACK_IN = 17
    BACK_OUT = 18


# ========== Fields That Can Target Groups ==========
# Used for trigger validation systems
TARGET_FIELDS: Final[tuple[str, ...]] = ("a51", "a71", "a401", "a395", "a76")
"""WARNING: Update the list in main.js !! (if editing)"""

# ============================================================================
# GAME CONSTANTS
# ============================================================================

class PowerLevel:
    """Level toggle group for power (shottype stuff)"""
    LEVEL_0 = 143
    LEVEL_1 = 144
    LEVEL_2 = 145
    LEVEL_3 = 146
    LEVEL_4 = 147

# Timing
TICK: Final[float] = 1 / 240  # One game tick in seconds

# Speed
PLR_SPEED: Final[float] = 311.58  # Player movement speed in studs/second

# Special Collision IDs
BOTTOM_BORDER = 12
RIGHT_BORDER = 8
TOP_BORDER = 9
LEFT_BORDER = 7

# Special Item IDs
LIVES = 15
POWER_LEVEL = 12
BOMB_COUNTER = 11
SCORE = 16

# Special Groups
PLR: Final[int] = 2
"""Player group and collision ID"""
SCREEN_CENTER: Final[int] = 30
"""Center group of 1200 x 900 screen window"""
GAME_CENTER: Final[int] = 109
"""Center group of 360 x 420 game window"""
GAME_BOTTOM_LEFT: Final[int] = 110
"""Bottom-left corner of 360 x 420 game window"""
NORTH_GROUP: Final[int] = 26
"""North of the map. Used for resetting rotation"""
DEBUG_UI_GROUP: Final[int] = 33
"""Group for debug viewing hidden objects (e.g. pointers, item counters)"""

PICKUP_RANDOM_MOVE = 129

HOMING_TRACKER_OBJ = 130
HOMING_TRACKER_BLOCK = 133
HOMING_TRACKER_COL_ID = 14

# Empty Groups for Remapping   (e.g. 21 => group)
# Holder object for empty groups is directly placed on the physical player, layer 0
EMPTY1: Final[int] = 21
EMPTY2: Final[int] = 23
EMPTY_EMITTER: Final[int] = 24
EMPTY_BULLET: Final[int] = 10
EMPTY_COLLISION: Final[int] = 104
EMPTY_TARGET_GROUP: Final[int] = 20

EMPTY_MULTITARGET: Final[int] = 25 # Restricted; multitarget exclusive

# Distance
OFFSCREEN_DIST: Final[int] = 480  # Minimum distance to get bullet offscreen
MIN_ANGLE: Final[int] = 3  # Minimum angle for GuiderCircles

EDITOR_LAYER: Final[int] = 4
"""Default editor layer"""

RESTRICTED_GROUPS: Final[tuple[int, ...]] = (
    1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 19, 22, 9999
)
"""Reserved groups for various safety reasons - DO NOT USE THESE GROUP IDs"""

# ============================================================================
# SPEED PROFILES
# ============================================================================

# Predefined speed profiles for bullet movement.
# Time to travel OFFSCREEN_DIST (480 studs) at different speeds.
SPEED_VERY_SLOW: Final[float] = 480 / 32  # 15.0 seconds
SPEED_SLOW: Final[float] = 480 / 16       # 7.5 seconds
SPEED_MEDIUM: Final[float] = 480 / 8      # 3.75 seconds
SPEED_FAST: Final[float] = 480 / 4        # 1.875 seconds
SPEED_VERY_FAST: Final[float] = 480 / 2   # 0.9375 seconds
