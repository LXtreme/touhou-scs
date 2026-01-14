"""
Touhou SCS - Component Module

Component class for building trigger sequences with method chaining.

URGENT: SpellBuilder is not yet implemented! stuff is commented out until then.
"""

from __future__ import annotations
from contextlib import contextmanager
import functools
from typing import Any, Callable, NamedTuple

from touhou_scs import enums as enum, lib, utils as util
from touhou_scs.utils import unknown_g, warn
from touhou_scs.types import Trigger


_RESTRICTED_LOOKUP = { group_id: True for group_id in enum.RESTRICTED_GROUPS }

ppt = enum.Properties # shorthand

class ScaleSettings(NamedTuple):
    factor: float
    hold: float
    duration: float
    type: int
    rate: float
    reverse: bool

scale_keyframes: dict[ScaleSettings, Component] = {}

@functools.lru_cache(maxsize=4096)
def _validate_params_cached(*,
    positive: tuple[float | int, ...] | None = None,
    non_negative: tuple[float | int, ...] | None = None,
    targets: int | tuple[int, ...] | None = None,
    type: int | None = None,
    rate: float | None = None,
    factor: float | None = None,
    item_id: int | None = None
) -> None:
    """Internal cached validation function. All sequences must be tuples."""
    
    if positive is not None:
        for val in positive:
            if val <= 0:
                raise ValueError(f"Values must be positive (>0). Got: {list(positive)}")
    
    if non_negative is not None:
        for val in non_negative:
            if val < 0:
                raise ValueError(f"Values must be non-negative (>=0). Got: {list(non_negative)}")
    
    if targets is not None:
        if isinstance(targets, int):
            if targets == -1:
                raise RuntimeError(f"Trigger requires an active target context. Got: {targets}")
            if targets <= 0:
                raise ValueError(f"Target Group '{targets}' must be positive (>0).")
            if targets in _RESTRICTED_LOOKUP:
                raise ValueError(f"Target Group '{targets}' is restricted.")
        else:
            for g in targets:
                if g == -1:
                    raise RuntimeError(f"Trigger requires an active target context. Got: {g}")
                if g <= 0:
                    raise ValueError(f"Target Group '{g}' must be positive (>0).")
                if g in _RESTRICTED_LOOKUP:
                    raise ValueError(f"Target Group '{g}' is restricted.")
    
    if type is not None and (not (0 <= type <= 18) or not type.is_integer()):
        raise ValueError(f"Easing 'type' must be an int in range 0-18. Got: {type}")
    if rate is not None and not (0.10 < rate <= 20.0):
        raise ValueError(f"Easing 'rate' must be in range 0.10-20.0, Got: {rate}")
    if factor is not None and factor <= 0:
        raise ValueError(f"Factor must be >0. Got: {factor}")
    if factor == 1:
        raise ValueError("Factor of multiplying/dividing by 1 has no effect")
    if item_id is not None and not (1 <= item_id <= 9999):
        raise ValueError(f"Item ID must be a positive int in range 1-9999. Got: {item_id}")

def validate_params(*,
    positive: float | int | list[float | int] | None = None,
    non_negative: float | int | list[float | int] | None = None,
    targets: int | list[int] | None = None,
    type: int | None = None,
    rate: float | None = None,
    factor: float | None = None,
    item_id: int | None = None
) -> None:
    """Validates common trigger parameters. Converts lists to tuples for caching."""
    
    # Convert to tuples for hashability
    positive_tuple: tuple[float | int, ...] | None = None
    if isinstance(positive, list):
        positive_tuple = tuple(positive)
    elif isinstance(positive, (int, float)):
        positive_tuple = (positive,)
    
    non_negative_tuple: tuple[float | int, ...] | None = None
    if isinstance(non_negative, list):
        non_negative_tuple = tuple(non_negative)
    elif isinstance(non_negative, (int, float)):
        non_negative_tuple = (non_negative,)
    
    targets_tuple: int | tuple[int, ...] | None = None
    if isinstance(targets, list):
        targets_tuple = tuple(targets)
    elif isinstance(targets, int):
        targets_tuple = targets
    
    _validate_params_cached(
        positive=positive_tuple,
        non_negative=non_negative_tuple,
        targets=targets_tuple,
        type=type,
        rate=rate,
        factor=factor,
        item_id=item_id
    )
    
    # Check counter bounds AFTER caching (counter is dynamic, can't be cached)
    if targets is not None:
        c = util.unknown_g.counter
        targets_list = [targets] if isinstance(targets, int) else targets
        
        for g in targets_list:
            if g > c:
                raise ValueError(f"Target Group '{g}' is out of valid range (1-{c}).")

def enforce_solid_groups(*groups: int):
    """Mark groups as solid (objects). Validation deferred until export."""
    for g in groups:
        lib.solid_groups_to_enforce.add(g)


class Component:
    def __init__(self, name: str, callerGroup: int, editorLayer: int = 4):
        self.name: str = name
        self.caller: int = callerGroup
        self.groups: list[int] = [callerGroup]
        self.editorLayer: int = editorLayer

        self.target: int = -1
        self.requireSpawnOrder: bool | None = None
        self.triggers: list[Trigger] = []
        self.current_pc: lib.GuiderCircle | None = None

        self._pointer: Pointer | None = None
        self._instant: InstantPatterns | None = None
        self._timed: TimedPatterns | None = None

        lib.all_components.append(self)

    @property
    def pointer(self):
        if self._pointer is None: self._pointer = Pointer(self)
        return self._pointer

    @property
    def instant(self):
        if self._instant is None: self._instant = InstantPatterns(self)
        return self._instant

    @property
    def timed(self):
        if self._timed is None: self._timed = TimedPatterns(self)
        return self._timed

    def get_triggers(self, trigger: dict[str, Any]) -> list[Trigger]:
        matches: list[Trigger] = []
        for t in self.triggers:
            num_matches = 0
            for key, value in trigger.items():
                if value is None:
                    raise ValueError("get_triggers: trigger property value cannot be None")
                if t.get(key) == value or value is Any:
                    num_matches += 1
            if num_matches == len(trigger): matches.append(t)
        return matches

    def has_trigger_properties(self, trigger: dict[str, Any]):
        if len(trigger) == 0:
            raise ValueError("has_trigger_properties: empty trigger dict given")
        return bool(self.get_triggers(trigger))

    def create_trigger(self, obj_id: int, x: float, target: int) -> Trigger:
        return Trigger({
            ppt.OBJ_ID: obj_id,
            ppt.X: x,
            ppt.TARGET: target,
            ppt.GROUPS: self.groups,
            ppt.EDITOR_LAYER: self.editorLayer,
            ppt.SPAWN_TRIGGERED: True,
            ppt.MULTI_TRIGGERED: True,
        })

    def assert_spawn_order(self, required: bool):
        self.requireSpawnOrder = required
        return self

    def _flatten_groups(self, *groups: int | list[int]) -> list[int]:
        result: list[int] = []
        for g in groups:
            if isinstance(g, list): result.extend(g)
            else: result.append(g)
        if len(result) != len(set(result)):
            raise ValueError(f"Flatten Groups: Duplicate groups found!: \n{result}")
        if len(result) == 0:
            raise ValueError("Flatten Groups: No groups provided!")
        return result

    @contextmanager
    def temp_context(self, target: int | None = None, groups: int | list[int] | None = None):
        """Temporarily set_context, then restore previous state."""
        old_target = self.target
        old_groups = self.groups.copy()

        try:
            self.set_context(target=target, groups=groups)
            yield self
        finally:
            self.target = old_target
            self.groups = old_groups

    def set_context(self, *,
        target: int | None = None, groups: int | list[int] | None = None):
        """Set a context for trigger target and groups. Either or both can be set."""
        if target is None and groups is None:
            raise ValueError("set_context: must provide target or groups")
        if target is not None:
            validate_params(targets=target)
            self.target = target
        if groups is not None:
            self.groups = [self.caller] + self._flatten_groups(groups)

        return self

    def clear_context(self, *, target_only: bool = False, groups_only: bool = False):
        """
        Clear active context. By default clears both.
            target_only: Clear only target context
            groups_only: Clear only groups context
        """
        no_param_given = not (target_only or groups_only)
        if target_only and groups_only:
            raise ValueError("clear_context: cannot use both target_only and groups_only")
        if target_only or no_param_given: self.target = -1
        if groups_only or no_param_given: self.groups = [self.caller]

        return self

    # ===========================================================
    #
    # TRIGGER METHODS
    #
    # ===========================================================

    def Spawn(self, time: float,
        target: int | Component, spawnOrdered: bool, *,
        remap: str | None = None, delay: float = 0, reset_remap: bool = False):
        """Spawn another component or group's triggers"""
        target = target.caller if isinstance(target, Component) else target
        validate_params(targets=target, non_negative=delay)

        trigger = self.create_trigger(enum.ObjectID.SPAWN, util.time_to_dist(time), target)

        if spawnOrdered: trigger[ppt.SPAWN_ORDERED] = True
        if delay > 0: trigger[ppt.SPAWN_DELAY] = delay
        if remap: _, trigger[ppt.REMAP_STRING] = util.translate_remap_string(remap)
        if reset_remap: trigger[ppt.RESET_REMAP] = True

        self.triggers.append(trigger)
        return self

    def Toggle(self, time: float, activateGroup: bool):
        """
        Activating does not spawn the target, it only enables it.
        WARNING: A deactivated object cannot be reactivated by a different group

        (collision triggers might be different)
        """
        validate_params(targets=self.target)

        trigger = self.create_trigger(enum.ObjectID.TOGGLE, util.time_to_dist(time), self.target)
        trigger[ppt.ACTIVATE_GROUP] = activateGroup

        self.triggers.append(trigger)
        return self

    def MoveTowards(self, time: float, targetDir: int, *,
        t: float, dist: int,
        type: int = 0, rate: float = 1.0, dynamic: bool = False):
        """Move target a set distance towards another group (direction mode)"""
        validate_params(targets=[self.target, targetDir], non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(enum.ObjectID.MOVE, util.time_to_dist(time), self.target)

        trigger[ppt.DURATION] = t
        trigger[ppt.MOVE_DIRECTION_MODE] = True
        trigger[ppt.MOVE_SMALL_STEP] = True
        trigger[ppt.MOVE_TARGET_DIR] = targetDir
        trigger[ppt.MOVE_TARGET_CENTER] = self.target
        trigger[ppt.MOVE_DIRECTION_MODE_DISTANCE] = dist
        trigger[ppt.EASING] = type
        trigger[ppt.EASING_RATE] = rate

        if dynamic: trigger[ppt.DYNAMIC] = True
        if t == 0: trigger[ppt.MOVE_SILENT] = True

        self.triggers.append(trigger)
        return self

    def Pulse(self, time: float,
        hsb: lib.HSB, *, exclusive: bool = False,
        fadeIn: float = 0, t: float = 0, fadeOut: float = 0):
        validate_params(non_negative=[fadeIn, t, fadeOut], targets=self.target)

        trigger = self.create_trigger(enum.ObjectID.PULSE, util.time_to_dist(time), self.target)

        trigger[ppt.PULSE_HSV] = True
        trigger[ppt.PULSE_TARGET_TYPE] = True
        #a0a0 for multiplicative, a1a1 for additive (its the checkbox for 's' and 'v')
        trigger[ppt.PULSE_HSV_STRING] = f"{hsb.h}a{hsb.s}a{hsb.b}a1a1"
        trigger[ppt.PULSE_FADE_IN] = fadeIn
        trigger[ppt.PULSE_HOLD] = t
        trigger[ppt.PULSE_FADE_OUT] = fadeOut
        trigger[ppt.PULSE_EXCLUSIVE] = exclusive

        self.triggers.append(trigger)
        return self

    def MoveBy(self, time: float, *,
        dx: float, dy: float,
        t: float = 0, type: int = 0, rate: float = 1.0):
        validate_params(targets=self.target, non_negative=t, type=type, rate=rate)

        trigger = self.create_trigger(enum.ObjectID.MOVE, util.time_to_dist(time), self.target)

        trigger[ppt.DURATION] = t
        trigger[ppt.MOVE_X] = dx
        trigger[ppt.MOVE_Y] = dy
        trigger[ppt.MOVE_SMALL_STEP] = True
        trigger[ppt.EASING] = type
        trigger[ppt.EASING_RATE] = rate

        if t == 0: trigger[ppt.MOVE_SILENT] = True

        self.triggers.append(trigger)
        return self

    def GotoGroup(self, time: float, location: int, *,
        t: float = 0, type: int = 0, rate: float = 1.0):
        validate_params(targets=[self.target, location], non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(enum.ObjectID.MOVE, util.time_to_dist(time), self.target)

        trigger[ppt.MOVE_TARGET_CENTER] = self.target
        trigger[ppt.MOVE_TARGET_LOCATION] = location
        trigger[ppt.MOVE_TARGET_MODE] = True
        trigger[ppt.DURATION] = t
        trigger[ppt.EASING] = type
        trigger[ppt.EASING_RATE] = rate

        if t == 0: trigger[ppt.MOVE_SILENT] = True

        self.triggers.append(trigger)
        return self

    def SetPosition(self, time: float, *, x: float, y: float):
        """
        2 Triggers instantly set target's position relative to origin.
            (bottom left of game window)
        """
        validate_params(targets=self.target)
        if self.requireSpawnOrder is not True:
            raise RuntimeError("SetPosition: Component must require spawn order.")
        
        self.GotoGroup(time, location=enum.GAME_BOTTOM_LEFT, t=0)
        self.MoveBy(time * enum.TICK*2, dx=x, dy=y)
        return self

    def Rotate(self, time: float, *,
        angle: float,
        center: int | None = None,
        t: float = 0, type: int = 0, rate: float = 1.0):
        """Rotate target by angle (degrees, clockwise is positive)"""
        if center is None: center = self.target
        validate_params(targets=[self.target, center], non_negative=t, type=type, rate=rate)

        trigger = self.create_trigger(enum.ObjectID.ROTATE, util.time_to_dist(time), self.target)

        trigger[ppt.ROTATE_CENTER] = center
        trigger[ppt.ROTATE_ANGLE] = angle
        trigger[ppt.DURATION] = t
        trigger[ppt.EASING] = type
        trigger[ppt.EASING_RATE] = rate

        self.triggers.append(trigger)
        return self

    def PointToGroup(self, time: float,
        targetDir: int, *,
        t: float = 0, type: int = 0, rate: float = 1.0, dynamic: bool = False):
        """Point target towards another group"""
        validate_params(targets=self.target, non_negative=t, type=type, rate=rate)
        enforce_solid_groups(targetDir)

        trigger = self.create_trigger(enum.ObjectID.ROTATE, util.time_to_dist(time), self.target)

        trigger[ppt.ROTATE_TARGET] = targetDir
        trigger[ppt.ROTATE_CENTER] = self.target
        trigger[ppt.ROTATE_AIM_MODE] = True
        trigger[ppt.DURATION] = t
        trigger[ppt.EASING] = type
        trigger[ppt.EASING_RATE] = rate
        trigger[ppt.DYNAMIC] = dynamic

        if dynamic and (type or rate != 1.0):
            raise ValueError(
                f"PointToGroup: dynamic aiming cannot use easing. \n"
                f"Given type {type}, rate {rate}")

        self.triggers.append(trigger)
        return self

    def Scale(self, time: float, *,
        factor: float, hold: float = 0, t: float = 0,
        type: int = 0, rate: float = 1.0, reverse: bool = False):
        """
        Scale target by a factor using Keyframes.

        't' is the time to scale and 'hold' is the time to stay at that scale.

        Reverse mode: Start at full size and scale down (doesnt use hold)
        Optional: t, hold, type, rate, reverse
        """
        validate_params(targets=self.target, factor=factor, non_negative=[t, hold], type=type, rate=rate)

        if hold and reverse:
            warn("Scale: 'hold' time is ignored in reverse mode: "
                "(no need to hold if it goes to original scale)")
        if not hold and not reverse:
            warn("Scale: 'hold' time is 0 but not in reverse."
                f" Target will instantly revert to full size after {t}s.")

        scale_settings = ScaleSettings(factor, hold, t, type, rate, reverse)

        if scale_settings in scale_keyframes:
            keyframe_group = scale_keyframes[scale_settings].caller
        else:
            name = f"Keyframe Scale<{factor}>,T<{t}>,Reverse<{reverse}>"
            new_keyframe_group = Component(name, unknown_g(), 6) \
                .assert_spawn_order(True)

            def keyframe_obj(*, scale: float, duration: float, order: int,
                close_loop: bool = False, ease_type: int = 0, ease_rate: float = 1.0):
                new_keyframe_group.triggers.append({ #type: ignore
                    ppt.OBJ_ID: enum.ObjectID.KEYFRAME_OBJ,
                    ppt.X: 0.0, ppt.Y: 0.0,
                    ppt.GROUPS: [new_keyframe_group.caller],
                    ppt.KEYFRAME_OBJ_MODE: 0,  # time mode
                    ppt.KEYFRAME_ID: new_keyframe_group.caller,
                    ppt.CLOSE_LOOP: close_loop,
                    ppt.SCALE: scale,
                    ppt.DURATION: duration,
                    ppt.ORDER_INDEX: order,
                    ppt.EASING: ease_type,
                    ppt.EASING_RATE: ease_rate,
                    ppt.LINE_OPACITY: 1.0,
                })

            if reverse:
                keyframe_obj(scale=1, duration=0, order=1)
                keyframe_obj(scale=factor, duration=t, order=2,
                    ease_type=type, ease_rate=rate, close_loop=True)
            else:
                keyframe_obj(scale=1, duration=t, order=1, ease_type=type, ease_rate=rate)
                keyframe_obj(scale=factor, duration=hold, order=2)
                keyframe_obj(scale=factor, duration=0, order=3, close_loop=True)

            keyframe_group = new_keyframe_group.caller
            scale_keyframes[scale_settings] = new_keyframe_group

        trigger = self.create_trigger(enum.ObjectID.KEYFRAME_ANIM, util.time_to_dist(time), self.target)

        trigger[ppt.KEYMAP_ANIM_GID] = keyframe_group
        trigger[ppt.KEYMAP_ANIM_TIME_MOD] = 1.0
        trigger[ppt.KEYMAP_ANIM_POS_X_MOD] = 1.0
        trigger[ppt.KEYMAP_ANIM_POS_Y_MOD] = 1.0
        trigger[ppt.KEYMAP_ANIM_ROT_MOD] = 1.0
        trigger[ppt.KEYMAP_ANIM_SCALE_X_MOD] = 1.0
        trigger[ppt.KEYMAP_ANIM_SCALE_Y_MOD] = 1.0

        self.triggers.append(trigger)
        return self

    def Follow(self, time: float, targetDir: int, *,
        t: float = 0, x_mod: float = 1.0, y_mod: float = 1.0):
        """Make target follow another group's movement"""
        validate_params(targets=self.target, non_negative=t)

        trigger = self.create_trigger(enum.ObjectID.FOLLOW, util.time_to_dist(time), self.target)

        trigger[ppt.FOLLOW_GROUP] = targetDir
        trigger[ppt.FOLLOW_X_MOD] = x_mod
        trigger[ppt.FOLLOW_Y_MOD] = y_mod
        trigger[ppt.DURATION] = t

        self.triggers.append(trigger)
        return self

    def Alpha(self, time: float, *, opacity: float, t: float = 0):
        """Change target's opacity from a range of 0-100 over time."""
        validate_params(targets=self.target, non_negative=t)
        if not (0 <= opacity <= 100):
            raise ValueError("Opacity must be between 0 and 100")

        trigger = self.create_trigger(enum.ObjectID.ALPHA, util.time_to_dist(time), self.target)

        trigger[ppt.OPACITY] = opacity / 100.0
        trigger[ppt.DURATION] = t

        self.triggers.append(trigger)
        return self

    def _stop_trigger_common(self,
        time: float, target: int | Component, option: int, useControlID: bool):
        target = target.caller if isinstance(target, Component) else target
        validate_params(targets=target)

        trigger = self.create_trigger(enum.ObjectID.STOP, util.time_to_dist(time), target)

        trigger[ppt.STOP_OPTION] = option # 0=Stop, 1=Pause, 2=Resume
        trigger[ppt.STOP_USE_CONTROL_ID] = useControlID

        self.triggers.append(trigger)

    def Stop(self, time: float, *, target: int | Component, useControlID: bool = False):
        """WARNING: Does not stop all triggers, but does stop Move, Rotate, Follow, Pulse, Alpha, Scale, Spawn."""
        self._stop_trigger_common(time, target, 0, useControlID)
        return self

    def Pause(self, time: float, *, target: int | Component, useControlID: bool = False):
        """WARNING: Does not pause all triggers, but does stop Move, Rotate, Follow, Pulse, Alpha, Scale, Spawn."""
        self._stop_trigger_common(time, target, 1, useControlID)
        return self

    def Resume(self, time: float, *, target: int | Component, useControlID: bool = False):
        """Resume target's paused triggers."""
        self._stop_trigger_common(time, target, 2, useControlID)
        return self

    def Collision(self, time: float, *,
        blockA: int, blockB: int, activateGroup: bool, onExit: bool = False):
        validate_params(targets=self.target)

        trigger = self.create_trigger(enum.ObjectID.COLLISION, util.time_to_dist(time), self.target)

        trigger[ppt.BLOCK_A] = blockA
        trigger[ppt.BLOCK_B] = blockB
        trigger[ppt.ACTIVATE_GROUP] = activateGroup
        if onExit: trigger[ppt.TRIGGER_ON_EXIT] = True

        self.triggers.append(trigger)
        return self

    def Count(self, time: float, *, item_id: int, count: int, activateGroup: bool):
        validate_params(targets=self.target, item_id=item_id)

        trigger = self.create_trigger(enum.ObjectID.COUNT, util.time_to_dist(time), self.target)

        trigger[ppt.ITEM_ID] = item_id
        trigger[ppt.COUNT_TARGET] = count
        trigger[ppt.ACTIVATE_GROUP] = activateGroup
        trigger[ppt.MULTI_ACTIVATE] = True

        self.triggers.append(trigger)
        return self

    def Pickup(self, time: float, *, item_id: int, count: int, override: bool):
        """Change an Item ID value by 'count' amount, or set to amount w/ 'override'"""
        validate_params(item_id=item_id)

        if count == 0: raise ValueError("Pickup: Count is 0 (no change)")

        trigger = self.create_trigger(enum.ObjectID.PICKUP, util.time_to_dist(time), 10)

        del trigger[ppt.TARGET] # type: ignore
        trigger[ppt.ITEM_ID] = item_id
        trigger[ppt.PICKUP_COUNT] = count
        trigger[ppt.PICKUP_OVERRIDE] = override
        trigger[ppt.PICKUP_MULTIPLY_DIVIDE] = 0

        self.triggers.append(trigger)
        return self

    def PickupModify(self, time: float, *, item_id: int, factor: float,
        multiply: bool = False, divide: bool = False):
        """Multiply/divide an Item ID value by 'factor' amount"""
        validate_params(item_id=item_id, factor=factor)

        if multiply and divide:
            raise ValueError("PickupModify: cannot both multiply and divide")
        if not multiply and not divide:
            raise ValueError("PickupModify: must specify multiply=True or divide=True")

        mode = 1 if multiply else 2 # multiply=1, divide=2

        trigger = self.create_trigger(enum.ObjectID.PICKUP, util.time_to_dist(time), 10)

        del trigger[ppt.TARGET] # type: ignore
        trigger[ppt.ITEM_ID] = item_id
        trigger[ppt.PICKUP_MULTIPLY_DIVIDE] = mode
        trigger[ppt.PICKUP_MODIFIER] = factor

        self.triggers.append(trigger)
        return self


# ===========================================================
#
# MULTITARGET CLASS
#
# ===========================================================

class Multitarget:
    """Make triggers effect multiple targets using a remap to components full of spawns."""

    _powers: list[int] = [1, 2, 4, 8, 16, 32, 64]
    _initialized: bool = False
    _binary_bases: dict[int, Component] = {}

    @classmethod
    def _get_binary_components(cls, num_targets: int, comp: Component) -> list[Component]:
        """Get the binary components needed to represent num_of_targets."""

        if any(t[ppt.OBJ_ID] == enum.ObjectID.SPAWN for t in comp.triggers):
            warn(f"Spawn limit: [{comp.name}] Multitarget components cannot have Spawn triggers")

        if not cls._initialized: cls._initialize_binary_bases()

        max_targets: int = 2 ** len(cls._powers) - 1
        if not (1 <= num_targets <= max_targets):
            raise ValueError(f"num_targets must be between 1 and {max_targets}. Got: {num_targets}")

        comps: list[Component] = []
        remaining = num_targets
        for power in cls._powers[::-1]:
            if remaining >= power:
                comps.append(cls._binary_bases[power])
                remaining -= power

        return comps

    @classmethod
    def _initialize_binary_bases(cls):
        if cls._initialized: raise RuntimeError("Multitarget binary bases already initialized")

        for power in cls._powers:
            component = Component(f"BinaryBase_{power}", unknown_g(), 4)
            component.assert_spawn_order(False)
            # To add support for more parameters, add a new empty group and follow the pattern
            num_emptys = 5
            for i in range(0, power * num_emptys, num_emptys):
                rb = (util.Remap()
                    .pair(enum.EMPTY_BULLET, i + 6001)
                    .pair(enum.EMPTY_TARGET_GROUP, i + 6002)
                    .pair(enum.EMPTY1, i + 6003)
                    .pair(enum.EMPTY_EMITTER, i + 6004)
                    .pair(enum.EMPTY_COLLISION, i + 6005))
                component.Spawn(0, enum.EMPTY_MULTITARGET, True, remap=rb.build())
            cls._binary_bases[power] = component

        max_targets: int = 2 ** len(cls._powers) - 1
        print(f"Multitarget: Initialized {len(cls._powers)} binary components, {max_targets} targets supported)")
        cls._initialized = True

    @classmethod
    def spawn_with_remap(cls, caller: Component, time: float, num_targets: int, comp: Component,
        remap_callback: Callable[[dict[int, int], util.Remap], None]
    ) -> None:
        """
        Spawn binary components with custom remap logic via callback.

        caller: Component that will spawn the multitarget components
        comp: Component that will be called multiple times
        remap_callback: Function that receives (remap_pairs, remap_builder)
         and should call remap_builder.pair() to map sources to actual resources.
        """
        for mt_comp in cls._get_binary_components(num_targets, comp):
            remap = util.Remap()
            for spawn_trigger in mt_comp.triggers:
                remap_string = spawn_trigger.get(ppt.REMAP_STRING, None)
                assert remap_string is not None
                remap_pairs, _ = util.translate_remap_string(remap_string)

                remap_callback(remap_pairs, remap)

            remap.pair(enum.EMPTY_MULTITARGET, comp.caller)
            caller.Spawn(time, mt_comp.caller, False,
                remap=remap.build(), reset_remap=False)


# ===========================================================
#
# PATTERN CLASSES
#
# ===========================================================

# Threshold for using multitarget optimization in CleanPointerCircle
# Batches smaller than this are processed manually to avoid edge case bugs
_MULTITARGET_THRESHOLD = 8

class _PointerCleanup:
    follow_comps: dict[float, Component] = {}
    _goto_comp_storage: Component | None = None
    
    @classmethod
    def get_goto_comp(cls) -> Component:
        if cls._goto_comp_storage is None:
            cls._goto_comp_storage = (Component("PointerCleanup_Goto", unknown_g(), 5)
                .assert_spawn_order(False)
                .set_context(target=enum.EMPTY_BULLET)
                    .GotoGroup(0, enum.EMPTY_TARGET_GROUP, t=0)
            )
        return cls._goto_comp_storage
    
    @classmethod
    def get_follow_comp(cls, duration: float) -> Component:
        for c in cls.follow_comps.keys():
            if abs(c - duration) < 0.09:
                duration = c
                break
        
        if duration not in cls.follow_comps:
            comp = (Component(f"PointerCleanup_Follow_{duration}", unknown_g(), 5)
                .assert_spawn_order(False)
                .set_context(target=enum.EMPTY_BULLET)
                    .Follow(0, enum.EMPTY_EMITTER, t=duration)
            )
            cls.follow_comps[duration] = comp
        return cls.follow_comps[duration]

class Pointer:
    def __init__(self, component: Component):
        self._component = component
        self._params: Any = ()

    @property
    def center(self) -> int:
        """Center of the active PointerCircle."""
        if self._component.current_pc is None:
            raise RuntimeError(f"Component '{self._component.name}' has no active pointer circle")
        return self._component.current_pc.center
    
    def point(self) -> int:
        """First pointer of the active PointerCircle."""
        if self._component.current_pc is None:
            raise RuntimeError(f"Component '{self._component.name}' has no active pointer circle")
        return self._component.current_pc.point

    def SetPointerCircle(self, time: float, *, location: int, duration: float = 0, set_north: bool = True):
        if self._component.current_pc is not None:
            raise RuntimeError("Pointer.SetPointerCircle: A PointerCircle is already active")
        if duration < 0:
            raise ValueError("Pointer.SetPointerCircle: duration cannot be negative")
        
        pc = lib.GuiderCircle(point=0, center=0)
        self._component.current_pc = pc
        self._params = (time, duration)
        
        with self._component.temp_context(target=lib.circle1.all):
            self._component.GotoGroup(time - enum.TICK*2, location)
            if set_north:
                self._component.PointToGroup(time - enum.TICK, enum.NORTH_GROUP)
        with self._component.temp_context(target=pc.center):
            self._component.GotoGroup(time, lib.circle1.center)
        
        return self._component

    def CleanPointerCircle(self):
        """Remove active Pointer-based GuiderCircle"""
        if self._component.current_pc is None:
            raise RuntimeError("Pointer.CleanPointerCircle: No active PointerCircle to clean")
        
        time, duration = self._params
        pc = self._component.current_pc
        c1 = lib.circle1
        
        used_pointers = [(g, c1.groups[i]) for i, g in enumerate(pc.groups) if g != -1]
        
        # === PHASE 1: Move pointers to circle1 positions (GotoGroup) ===
        pair_iter = iter(used_pointers)
        goto_comp = _PointerCleanup.get_goto_comp()
        
        def remap_goto(remap_pairs: dict[int, int], remap: util.Remap):
            pc_pointer, c1_pointer = next(pair_iter)
            for source, target in remap_pairs.items():
                if source == enum.EMPTY_BULLET:
                    remap.pair(target, pc_pointer)
                elif source == enum.EMPTY_TARGET_GROUP:
                    remap.pair(target, c1_pointer)
                else:
                    remap.pair(target, enum.EMPTY_MULTITARGET)
        
        remaining = len(used_pointers)
        while remaining > 0:
            batch_size = 64 if remaining > 127 else remaining
            
            # Skip small batches - process manually instead
            if batch_size < _MULTITARGET_THRESHOLD:
                for _ in range(batch_size):
                    pc_pointer, c1_pointer = next(pair_iter)
                    with self._component.temp_context(target=pc_pointer):
                        self._component.GotoGroup(time - enum.TICK, c1_pointer)
            else:
                Multitarget.spawn_with_remap(self._component, time - enum.TICK, batch_size, goto_comp, remap_goto)
            
            remaining -= batch_size
        
        # === PHASE 2: Add follow behavior ===
        if duration < 0:
            self._component.current_pc = None
            return self._component
        
        pair_iter = iter(used_pointers)
        follow_comp = _PointerCleanup.get_follow_comp(duration)
        
        def remap_follow(remap_pairs: dict[int, int], remap: util.Remap):
            pc_pointer, _ = next(pair_iter)
            for source, target in remap_pairs.items():
                if source == enum.EMPTY_BULLET:
                    remap.pair(target, pc_pointer)
                elif source == enum.EMPTY_EMITTER:
                    remap.pair(target, pc.center)
                else:
                    remap.pair(target, enum.EMPTY_MULTITARGET)
        
        remaining = len(used_pointers)
        while remaining > 0:
            batch_size = 64 if remaining > 127 else remaining
            
            # Skip small batches - process manually instead
            if batch_size < _MULTITARGET_THRESHOLD:
                for _ in range(batch_size):
                    pc_pointer, _ = next(pair_iter)
                    with self._component.temp_context(target=pc_pointer):
                        self._component.Follow(time, pc.center, t=duration)
            else:
                Multitarget.spawn_with_remap(self._component, time, batch_size, follow_comp, remap_follow)
            
            remaining -= batch_size
        
        self._component.current_pc = None
        return self._component


class InstantPatterns:
    def __init__(self, component: Component):
        self._component = component


    def Arc(self, time: float, comp: Component, bullet: lib.BulletPool, *,
        numBullets: int, angle: float, centerAt: float = 0, _radialBypass: bool = False):
        """
        Arc pattern - partial circle of bullets

        Component must use EMPTY_BULLET and EMPTY_TARGET_GROUP. \n
        Optional: centerAt
        """
        IA = "Instant Arc:"

        util.enforce_component_targets("Instant Arc", comp,
            requires={enum.EMPTY_BULLET, enum.EMPTY_TARGET_GROUP },
            excludes={enum.EMPTY_MULTITARGET}
        )
        
        # Validate angle and centerAt
        if not (0 < angle <= 360):
            raise ValueError(f"{IA} angle must be between 0 (exclusive) and 360 (inclusive). Got: {angle}")
        if not (0 <= centerAt < 360):
            raise ValueError(f"{IA} centerAt must be between 0 (inclusive) and 360 (exclusive). Got: {centerAt}")
        
        pc = self._component.current_pc
        if pc is None:
            raise RuntimeError(f"{IA} requires an active PointerCircle in the component.")

        startAngle = centerAt - angle / 2
        endAngle = centerAt + angle / 2
        groups = pc.angle_to_groups(startAngle, endAngle, numBullets, _radialBypass)
        iter_groups = iter(groups)

        def remap_arc(remap_pairs: dict[int, int], remap: util.Remap):
            nonlocal iter_groups
            bullet_group, bullet_col = bullet.next()
            pointer = next(iter_groups)
            for source, target in remap_pairs.items():
                if source == enum.EMPTY_BULLET:
                    remap.pair(target, bullet_group)
                elif source == enum.EMPTY_COLLISION:
                    remap.pair(target, bullet_col)
                elif source == enum.EMPTY_TARGET_GROUP:
                    remap.pair(target, pointer)
                elif source == enum.EMPTY_EMITTER:
                    remap.pair(target, pc.center)
                else:
                    remap.pair(target, enum.EMPTY_MULTITARGET)

        Multitarget.spawn_with_remap(self._component, time, numBullets, comp, remap_arc)

        return self._component


    def Radial(self, time: float, comp: Component, bullet: lib.BulletPool, *,
        numBullets: int | None = None, spacing: int | None = None, centerAt: float = 0):
        """
        Radial pattern - full 360° circle of bullets

        Component must use EMPTY_BULLET and EMPTY_TARGET_GROUP.  \n
        Optional: spacing or numBullets, centerAt
        """
        IR = "Instant Radial:"
        util.enforce_component_targets(IR, comp,
            requires={enum.EMPTY_BULLET, enum.EMPTY_TARGET_GROUP },
            excludes={enum.EMPTY_MULTITARGET}
        )

        if not (0 <= centerAt < 360):
            raise ValueError(f"{IR} centerAt must be between 0 (inclusive) and 360 (exclusive). Got: {centerAt}")

        if spacing and numBullets:
            if numBullets != int(360 / spacing):
                raise ValueError(f"{IR} spacing and numBullets don't match!\n\n"
                f"(numOfBullets should be {int(360 / spacing)}, \n"
                f"or spacing should be {int(360 / numBullets)}, \n\n"
                f"or just use one or the other)")
        elif spacing: numBullets = int(360 / spacing)
        elif numBullets: spacing = int(360 / numBullets)
        else: raise ValueError(f"{IR} must provide either spacing or numBullets")

        self.Arc(time, comp, bullet,
            numBullets=numBullets, angle=360, centerAt=centerAt, _radialBypass=True)

        return self._component

    def Line(self, time: float, comp: Component, emitter: int,
        targetDir: int, bullet: lib.BulletPool, *,
        numBullets: int, fastestTime: float, slowestTime: float, dist: int,
        type: int = 0, rate: float = 1.0):
        """
        Line pattern - builds MoveTowards triggers at different speeds, forming a line.

        Comp requires EMPTY_BULLET and EMPTY_MULTITARGET.
        Optional: type, rate
        """
        validate_params(positive=[fastestTime, slowestTime], type=type, rate=rate)
        IL = "Instant.Line:"

        util.enforce_component_targets(IL, comp,
            requires={ enum.EMPTY_BULLET, enum.EMPTY_EMITTER },
            excludes={ enum.EMPTY_TARGET_GROUP, enum.EMPTY_MULTITARGET,
                enum.EMPTY1, enum.EMPTY2 }
        )

        if bullet.has_orientation and not comp.has_trigger_properties({ppt.ROTATE_AIM_MODE:Any}):
            warn(f"{IL} Bullet has orientation enabled, but component has no PointToGroup trigger. Bullets may not face the correct direction.")

        if fastestTime >= slowestTime:
            raise ValueError(f"{IL} slowestTime {slowestTime} must be greater than fastestTime {fastestTime}")
        if numBullets < 3:
            raise ValueError(f"{IL} numBullets must be at least 3. Got: {numBullets}")

        bullet_groups: list[int] = []

        def remap_line(remap_pairs: dict[int, int], remap: util.Remap):
            bullet_group, bullet_col = bullet.next()
            for source, target in remap_pairs.items():
                if source == enum.EMPTY_BULLET:
                    bullet_groups.append(bullet_group)
                    remap.pair(target, bullet_group)
                elif source == enum.EMPTY_COLLISION:
                    remap.pair(target, bullet_col)
                elif source == enum.EMPTY_EMITTER:
                    remap.pair(target, emitter)
                else:
                    remap.pair(target, enum.EMPTY_MULTITARGET)

        Multitarget.spawn_with_remap(self._component, time, numBullets, comp, remap_line)

        step = (slowestTime - fastestTime) / (numBullets - 1)
        for i, bullet_group in enumerate(bullet_groups):
            travel_time = fastestTime + step * i
            with self._component.temp_context(target=bullet_group):
                self._component.MoveTowards(
                    time, targetDir,
                    t=travel_time, dist=dist, type=type, rate=rate
                )

        return self._component

    # More pattern methods will be added here


class TimedPatterns:
    def __init__(self, component: Component):
        self._component = component

    def RadialWave(self, time: float, comp: Component, bullet: lib.BulletPool, *,
        waves: int, interval: float = 0, numBullets: int | None = None, spacing: int | None = None, centerAt: float = 0):
        """
        Radial Wave pattern - multiple waves of radial bullets over time

        Optional: spacing or numBullets, centerAt
        """
        RW = "RadialWave:"
        if waves < 1:
            raise ValueError(f"{RW} waves must be at least 1. Got: {waves}")
        elif waves == 1:
            raise ValueError(f"{RW} for single wave, use instant.Radial() instead")
        if interval < 0:
            raise ValueError(f"{RW} interval must be non-negative. Got: {interval}")

        for wave_number in range(waves):
            self._component.instant.Radial(
                time + (wave_number * interval),
                comp, bullet, numBullets=numBullets, spacing=spacing, centerAt=centerAt
            )

        return self._component

    def Line(self, time: float, comp: Component, targetDir: int, bullet: lib.BulletPool, *,
        numBullets: int, spacing: float, t: float, dist: int, type: int = 0, rate: float = 1.0):
        """
        Line pattern - bullets in a line over time with equal gaps between them

        Optional: type, rate
        """
        TL = "Timed.Line:"

        util.enforce_component_targets(TL, comp,
            requires={ enum.EMPTY_BULLET },
            excludes={ enum.EMPTY_TARGET_GROUP, enum.EMPTY_MULTITARGET,
                enum.EMPTY1, enum.EMPTY2, enum.EMPTY_EMITTER }
        )

        if bullet.has_orientation and not comp.has_trigger_properties({ppt.ROTATE_AIM_MODE:Any}):
            warn(f"{TL} Bullet has orientation enabled, but component has no PointToGroup triggers. Bullets may not face the correct direction.")

        if numBullets < 2:
            raise ValueError(f"{TL} numBullets must be at least 2. Got: {numBullets}")
        if spacing < 0:
            raise ValueError(f"{TL} spacing must be non-negative. Got: {spacing}")
        if t < 0:
            raise ValueError(f"{TL} t must be non-negative. Got: {t}")

        for i in range(0, numBullets):
            b, _ = bullet.next()
            self._component.Spawn(
                time + (i * spacing), comp.caller, True, remap=f"{enum.EMPTY_BULLET}:{b}")
            with self._component.temp_context(target=b):
                self._component.MoveTowards(
                    time + (i * spacing), targetDir, t=t, dist=dist, type=type, rate=rate
                )
        return self._component

    # More pattern methods will be added here
