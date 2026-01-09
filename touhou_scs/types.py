"""
Touhou SCS - Type Definitions
"""

from typing import Required, TypedDict, Protocol, Any

# ==========================================
# TRIGGER STRUCTURE
# ==========================================

Trigger = TypedDict("Trigger",{

    "1": Required[int],         # OBJ_ID - Trigger type
    "2": Required[float],       # X - X position (time-based)
    "20": Required[int],        # EDITOR_LAYER
    "51": Required[int],        # TARGET
    "57": Required[list[int]],  # GROUPS - Caller group(s)
    "62": Required[bool],       # SPAWN_TRIGGERED
    "87": Required[bool],       # MULTI_TRIGGERED
    # General optional fields
    "3": float,                 # Y - Y position (set by spread)
    "10": float,                # DURATION
    "30": int,                  # EASING
    "61": int,                  # EDITOR_LAYER_2
    "85": float,                # EASING_RATE
    "397": bool,                # DYNAMIC
    # Alpha
    "35": float,                # OPACITY
    # Follow
    "71": int,                  # FOLLOW_GROUP / ROTATE_CENTER / SCALE_CENTER, etc.
    "72": float,                # FOLLOW_X_MOD
    "73": float,                # FOLLOW_Y_MOD
    # Stop
    "580": int,                 # STOP_OPTION
    "535": bool,                # STOP_USE_CONTROL_ID
    # Keyframe Animate
    "76": int,                  # KEYMAP_ANIM_GID
    "520": float,               # KEYMAP_ANIM_TIME_MOD
    "521": float,               # KEYMAP_ANIM_POS_X_MOD
    "545": float,               # KEYMAP_ANIM_POS_Y_MOD
    "522": float,               # KEYMAP_ANIM_ROT_MOD
    "523": float,               # KEYMAP_ANIM_SCALE_X_MOD
    "546": float,               # KEYMAP_ANIM_SCALE_Y_MOD
    # Toggle
    "56": bool,                 # ACTIVATE_GROUP
    # Count
    "77": int,                  # COUNT_TARGET, PICKUP_COUNT
    "104": bool,                # MULTI_ACTIVATE
    # Pickup
    "139": bool,                # PICKUP_OVERRIDE
    "88": int,                  # PICKUP_MULTIPLY_DIVIDE
    "449": float,               # PICKUP_MODIFIER
    # Collision
    "80": int,                  # BLOCK_A (also ITEM_ID)
    "95": int,                  # BLOCK_B
    "93": bool,                 # TRIGGER_ON_EXIT
    # Pulse
    "45": float,                # PULSE_FADE_IN
    "46": float,                # PULSE_HOLD
    "47": float,                # PULSE_FADE_OUT
    "48": bool,                 # PULSE_HSV
    "49": str,                  # PULSE_HSV_STRING
    "52": bool,                 # PULSE_TARGET_TYPE
    "86": bool,                 # PULSE_EXCLUSIVE
    # Scale
    "150": float,               # SCALE_X
    "151": float,               # SCALE_Y
    "153": bool,                # SCALE_DIV_BY_X
    "154": bool,                # SCALE_DIV_BY_Y
    # Rotate
    "68": float,                # ROTATE_ANGLE
    "401": int,                 # ROTATE_TARGET
    "100": bool,                # ROTATE_AIM_MODE (also MOVE_TARGET_MODE)
    "403": int,                 # ROTATE_DYNAMIC_EASING
    # Spawn
    "442": str,                 # REMAP_STRING
    "581": bool,                # RESET_REMAP
    "441": bool,                # SPAWN_ORDERED
    "63": float,                # SPAWN_DELAY
    # Move
    "28": float,                # MOVE_X
    "29": float,                # MOVE_Y
    "393": bool,                # MOVE_SMALL_STEP
    "395": int,                 # MOVE_TARGET_CENTER
    "394": bool,                # MOVE_DIRECTION_MODE
    "396": float,               # MOVE_DIRECTION_MODE_DISTANCE
    "544": bool,                # MOVE_SILENT
}, total=False)


# ==========================================
# COMPONENT PROTOCOL
# ==========================================

class ComponentProtocol(Protocol):
    """
    Interface for Component objects.

    Any class implementing these attributes/methods can be used as a Component.
    This is Python's way of doing duck typing with type safety.
    """
    name: str
    caller: int
    groups: list[int]
    editorLayer: int
    requireSpawnOrder: bool | None
    triggers: list[Trigger]
    target: int
    current_pc: Any

    def assert_spawn_order(self, required: bool) -> "ComponentProtocol":
        """Set spawn order requirement. Returns self for chaining."""
        ...


# ==========================================
# SPELL PROTOCOL
# ==========================================

class SpellProtocol(Protocol):
    """Interface for Spell objects"""
    spell_name: str
    caller_group: int
    components: list[ComponentProtocol]

    def add_component(self, component: ComponentProtocol) -> "SpellProtocol":
        """Add component to spell. Returns self for chaining."""
        ...


# ==========================================
# RANDOM TYPE ALIASES AND STUFF
# ==========================================

TriggerArea = TypedDict('TriggerArea', {
    "min_x": int,
    "min_y": int,
    "max_x": int,
    "max_y": int
})

GroupID = int
"""Normal group (0-9999) or unknown_g placeholder (10000+)"""

Time = float
"""Time in seconds (positional arg in most triggers)"""

Distance = float
"""Distance in studs"""

Angle = float
"""Angle in degrees"""
