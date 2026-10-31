"""
Touhou SCS - Component Module

Component class for building trigger sequences with method chaining.
"""

from __future__ import annotations
from contextlib import contextmanager
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

from gmdbuilder.object_types import EASING, STOP_MODE
from gmdkit.models.prop.color import HSV

from touhou_scs.movements import CurveType, apply_bezier_movement
from touhou_scs import enums as enum, lib, utils as util
from touhou_scs.utils import unknown_g, warn
from touhou_scs.types import GenericObj
from touhou_scs.comp_extra import RandomPatterns

from gmdbuilder import ObjectType, from_object_string, new_obj, obj_id, obj_prop

ppt = obj_prop.Trigger

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
            if targets < 0:
                raise ValueError(f"Target Group '{targets}' must be non-negative (>=0).")
            if targets in enum.RESTRICTED_GROUPS:
                raise ValueError(f"Target Group '{targets}' is restricted.")
        else:
            for g in targets:
                if g == -1:
                    raise RuntimeError(f"Trigger requires an active target context. Got: {g}")
                if g < 0:
                    raise ValueError(f"Target Group '{g}' must be non-negative (>=0).")
                if g in enum.RESTRICTED_GROUPS:
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
    positive: float | list[float | int] | None = None,
    non_negative: float | list[float | int] | None = None,
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
    
    # Check group ID is within GD's hard limit
    if targets is not None:
        targets_list = [targets] if isinstance(targets, int) else targets
        for g in targets_list:
            if not (1 <= g <= 9999):
                raise ValueError(f"Target Group '{g}' is out of valid range (1-9999).")

def enforce_solid_groups(*groups: int):
    """Mark groups as solid (objects). Validation deferred until export."""
    for g in groups:
        lib.solid_groups_to_enforce.add(g)
    # Uncomment if needed:
    # lib._validate_solid_groups(*groups) #type: ignore


class Component:
    def __init__(self, name: str, callerGroup: int, editorLayer: int = 4):
        self.name: str = name
        self.caller: int = callerGroup
        self.groups: set[int] = { callerGroup }
        self.editorLayer: int = editorLayer

        self.target: int = -1
        self.requireSpawnOrder: bool | None = None
        self.triggers: list[GenericObj] = []
        self.current_pc: lib.GuiderCircle | None = None

        self._pointer: Pointer | None = None
        self._instant: InstantPatterns | None = None
        self._timed: TimedPatterns | None = None
        self._random: RandomPatterns | None = None

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

    @property
    def random(self) -> RandomPatterns:
        if self._random is None: self._random = RandomPatterns(self)
        return self._random

    def get_triggers(self, trigger: dict[str, Any]) -> list[GenericObj]:
        matches: list[GenericObj] = []
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

    def _register_trigger(self, trigger: GenericObj, x: float) -> GenericObj:
        """Stamp component-level properties onto a trigger and append it. Returns the trigger."""
        trigger[obj_prop.X] = x
        trigger[obj_prop.GROUPS] = set(self.groups) # copy: don't share one set across triggers
        trigger[obj_prop.EDITOR_L1] = self.editorLayer
        trigger[ppt.SPAWN_TRIGGER] = True
        trigger[ppt.MULTI_TRIGGER] = True
        self.triggers.append(trigger)
        return trigger

    def create_trigger(self, obj_id: int, x: float) -> GenericObj:
        """Create a trigger, register it with the component, and return it.
        Target fields are NOT set: assign the trigger-specific target (a51 etc.) at the call site."""
        return self._register_trigger(new_obj(obj_id), x) # type: ignore

    def assert_spawn_order(self, required: bool):
        self.requireSpawnOrder = required
        return self

    def _flatten_groups(self, *groups: int | set[int]) -> set[int]:
        """Flattens nested groups into a single set of ints. Raises on duplicates."""
        seen: set[int] = set()

        def walk(items: Any) -> None:
            for g in items:
                if isinstance(g, int):
                    if g in seen:
                        raise ValueError(f"Flatten Groups: Duplicate group {g} found!")
                    seen.add(g)
                else:
                    walk(g)

        walk(groups)
        return seen

    @contextmanager
    def temp_context(self, target: int | None = None, groups: int | set[int] | None = None,
        include_caller: bool = True):
        """Temporarily set_context, then restore previous state."""
        old_target = self.target
        old_groups = self.groups.copy()

        try:
            self.set_context(target=target, groups=groups, include_caller=include_caller)
            yield self
        finally:
            self.target = old_target
            self.groups = old_groups

    def set_context(self, *,
        target: int | None = None, groups: int | set[int] | None = None,
        include_caller: bool = True):
        """
        Set a context for trigger target and groups. Either or both can be set.
        include_caller=False makes triggers belong only to 'groups' (not spawned with the
        component itself), e.g. branch groups spawned by an ItemCompare.
        """
        if target is None and groups is None:
            raise ValueError("set_context: must provide target or groups")
        if target is not None:
            validate_params(targets=target)
            self.target = target
        if groups is not None:
            flat = self._flatten_groups(groups)
            self.groups = ({self.caller} | flat) if include_caller else flat

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
        if groups_only or no_param_given: self.groups = {self.caller}

        return self

    # ===========================================================
    #
    # TRIGGER METHODS
    #
    # ===========================================================

    def Spawn(self, time: float,
        target: int | Component, spawnOrdered: bool, *,
        remap: dict[int,int] | None = None, delay: float = 0, reset_remap: bool = False):
        """Spawn another component or group's triggers"""
        target = target.caller if isinstance(target, Component) else target
        validate_params(targets=target, non_negative=delay)

        trigger = self.create_trigger(obj_id.Trigger.SPAWN, util.time_to_dist(time))

        trigger[ppt.Spawn.TARGET_ID] = target

        if spawnOrdered: trigger[ppt.Spawn.ORDERED] = True
        if delay > 0: trigger[ppt.Spawn.DELAY] = delay
        if remap: trigger[ppt.Spawn.REMAPS] = remap
        if reset_remap: trigger[ppt.Spawn.RESET_REMAP] = True

        return self

    def Toggle(self, time: float, activateGroup: bool):
        """
        Activating does not spawn the target, it only enables it.
        WARNING: A deactivated object cannot be reactivated by a different group

        (collision triggers might be different)
        """
        validate_params(targets=self.target)

        trigger = self.create_trigger(obj_id.Trigger.TOGGLE, util.time_to_dist(time))

        trigger[ppt.Toggle.TARGET_ID] = self.target
        trigger[ppt.Toggle.ACTIVATE_GROUP] = activateGroup

        return self

    def MoveTowards(self, time: float, targetDir: int, *,
        t: float, dist: int,
        type: EASING = 0, rate: float = 1.0, dynamic: bool = False):
        """Move target a set distance towards another group (direction mode)"""
        validate_params(targets=[self.target, targetDir], non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(obj_id.Trigger.MOVE, util.time_to_dist(time))

        trigger[ppt.Move.TARGET_ID] = self.target

        trigger[ppt.Move.DURATION] = t
        trigger[ppt.Move.DIRECTION_MODE] = True
        trigger[ppt.Move.USE_SMALL_STEP] = True
        trigger[ppt.Move.TARGET_POS] = targetDir
        trigger[ppt.Move.TARGET_CENTER_ID] = self.target
        trigger[ppt.Move.TARGET_DISTANCE] = dist
        trigger[ppt.Move.EASING] = type
        trigger[ppt.Move.EASE_RATE] = rate

        if dynamic: trigger[ppt.Move.DYNAMIC_MODE] = True
        if t == 0: trigger[ppt.Move.SILENT] = True

        return self

    def Pulse(self, time: float, hsb: lib.HSB, *, 
        exclusive: bool = False, fadeIn: float = 0, t: float = 0, fadeOut: float = 0):
        validate_params(non_negative=[fadeIn, t, fadeOut], targets=self.target)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(obj_id.Trigger.PULSE, util.time_to_dist(time))

        trigger[ppt.Pulse.TARGET_ID] = self.target
        
        converted = HSV()
        converted.hue = hsb.h
        converted.saturation = hsb.s
        converted.value = hsb.b
        converted.saturation_add = True
        converted.value_add = True

        trigger[ppt.Pulse.USE_HSV] = True
        trigger[ppt.Pulse.TARGET_TYPE] = True
        #a0a0 for multiplicative, a1a1 for additive (its the checkbox for 's' and 'v')
        trigger[ppt.Pulse.HSV] = converted
        trigger[ppt.Pulse.FADE_IN] = fadeIn
        trigger[ppt.Pulse.HOLD] = t
        trigger[ppt.Pulse.FADE_OUT] = fadeOut
        trigger[ppt.Pulse.EXCLUSIVE] = exclusive

        return self

    def MoveBy(self, time: float, *, dx: float, dy: float,
        t: float = 0, type: EASING = 0, rate: float = 1.0):
        validate_params(targets=self.target, non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(obj_id.Trigger.MOVE, util.time_to_dist(time))

        trigger[ppt.Move.TARGET_ID] = self.target

        trigger[ppt.Move.DURATION] = t
        trigger[ppt.Move.MOVE_X] = dx
        trigger[ppt.Move.MOVE_Y] = dy
        trigger[ppt.Move.USE_SMALL_STEP] = True
        trigger[ppt.Move.EASING] = type
        trigger[ppt.Move.EASE_RATE] = rate

        if t == 0: trigger[ppt.Move.SILENT] = True

        return self

    def GotoGroup(self, time: float, location: int, *,
        t: float = 0, type: EASING = 0, rate: float = 1.0, dynamic: bool = False):
        validate_params(targets=[self.target, location], non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target, location)

        trigger = self.create_trigger(obj_id.Trigger.MOVE, util.time_to_dist(time))

        trigger[ppt.Move.TARGET_ID] = self.target

        trigger[ppt.Move.TARGET_CENTER_ID] = self.target
        trigger[ppt.Move.TARGET_POS] = location
        trigger[ppt.Move.TARGET_MODE] = True
        trigger[ppt.Move.DURATION] = t
        trigger[ppt.Move.EASING] = type
        trigger[ppt.Move.EASE_RATE] = rate

        if dynamic: trigger[ppt.Move.DYNAMIC_MODE] = True
        if t == 0: trigger[ppt.Move.SILENT] = True

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

    def Rotate(self, time: float, *, angle: float, center: int | None = None,
        t: float = 0, type: EASING = 0, rate: float = 1.0):
        """Rotate target by angle (degrees, clockwise is positive)"""
        if center is None: center = self.target
        validate_params(targets=[self.target, center], non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target)

        trigger = self.create_trigger(obj_id.Trigger.ROTATE, util.time_to_dist(time))

        trigger[ppt.Rotate.TARGET_ID] = self.target

        trigger[ppt.Rotate.CENTER_ID] = center
        trigger[ppt.Rotate.DEGREES] = angle
        trigger[ppt.Rotate.DURATION] = t
        trigger[ppt.Rotate.EASING] = type
        trigger[ppt.Rotate.EASE_RATE] = rate

        return self

    def PointToGroup(self, time: float, targetDir: int, *,
        t: float = 0, type: EASING = 0, rate: float = 1.0, dynamic: bool = False):
        """Point target towards another group"""
        validate_params(targets=self.target, non_negative=t, type=type, rate=rate)
        enforce_solid_groups(self.target, targetDir)

        trigger = self.create_trigger(obj_id.Trigger.ROTATE, util.time_to_dist(time))

        trigger[ppt.Rotate.TARGET_ID] = self.target

        trigger[ppt.Rotate.AIM_TARGET_ID] = targetDir
        trigger[ppt.Rotate.CENTER_ID] = self.target
        trigger[ppt.Rotate.AIM_MODE] = True
        trigger[ppt.Rotate.DURATION] = t
        trigger[ppt.Rotate.EASING] = type
        trigger[ppt.Rotate.EASE_RATE] = rate
        trigger[ppt.Rotate.DYNAMIC_MODE] = dynamic

        if dynamic and (type or rate != 1.0):
            raise ValueError(
                f"PointToGroup: dynamic aiming cannot use easing. \n"
                f"Given type {type}, rate {rate}")

        return self

    def Scale(self, time: float, *,
        factor: float, hold: float = 0, t: float = 0,
        type: EASING = 0, rate: float = 1.0, reverse: bool = False):
        """
        Scale target by a factor using Keyframes.

        't' is the time to scale and 'hold' is the time to stay at that scale.

        Reverse mode: Start at full size and scale down (doesnt use hold)
        Optional: t, hold, type, rate, reverse
        """
        validate_params(targets=self.target, factor=factor, non_negative=[t, hold], type=type, rate=rate)
        enforce_solid_groups(self.target)

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
                close_loop: bool = False, ease_type: EASING = 0, ease_rate: float = 1.0):
                new_keyframe_group.triggers.append({ # type: ignore
                    obj_prop.ID: obj_id.Trigger.KEYFRAME,
                    obj_prop.X: 0.0, obj_prop.Y: 0.0,
                    obj_prop.GROUPS: {new_keyframe_group.caller},
                    ppt.Keyframe.TIME_MODE: 0,  # time mode
                    ppt.Keyframe.KEY_ID: new_keyframe_group.caller,
                    ppt.Keyframe.CLOSE_LOOP: close_loop,
                    obj_prop.OLD_SCALE: scale,
                    ppt.Keyframe.DURATION: duration,
                    ppt.Keyframe.INDEX: order,
                    ppt.Keyframe.EASING: ease_type,
                    ppt.Keyframe.EASE_RATE: ease_rate,
                    ppt.Keyframe.LINE_OPACITY: 1.0,
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

        trigger = self.create_trigger(obj_id.Trigger.ANIMATE_KEYFRAME, util.time_to_dist(time))

        trigger[ppt.AnimateKeyframe.TARGET_ID] = self.target

        trigger[ppt.AnimateKeyframe.ANIMATION_ID] = keyframe_group
        trigger[ppt.AnimateKeyframe.TIME_MOD] = 1.0
        trigger[ppt.AnimateKeyframe.POS_X_MOD] = 1.0
        trigger[ppt.AnimateKeyframe.POS_Y_MOD] = 1.0
        trigger[ppt.AnimateKeyframe.ROTATION_MOD] = 1.0
        trigger[ppt.AnimateKeyframe.SCALE_X_MOD] = 1.0
        trigger[ppt.AnimateKeyframe.SCALE_Y_MOD] = 1.0

        return self

    def Follow(self, time: float, targetDir: int, *,
        t: float = 0, x_mod: float = 1.0, y_mod: float = 1.0):
        """Make target follow another group's movement"""
        validate_params(targets=self.target, non_negative=t)
        enforce_solid_groups(self.target, targetDir)

        trigger = self.create_trigger(obj_id.Trigger.FOLLOW, util.time_to_dist(time))

        trigger[ppt.Follow.TARGET_ID] = self.target

        trigger[ppt.Follow.FOLLOW_TARGET] = targetDir
        trigger[ppt.Follow.MOD_X] = x_mod
        trigger[ppt.Follow.MOD_Y] = y_mod
        trigger[ppt.Follow.DURATION] = t

        return self

    def Alpha(self, time: float, *, opacity: float, t: float = 0):
        """Change target's opacity from a range of 0-100 over time."""
        validate_params(targets=self.target, non_negative=t)
        enforce_solid_groups(self.target)
        if not (0 <= opacity <= 100):
            raise ValueError("Opacity must be between 0 and 100")

        trigger = self.create_trigger(obj_id.Trigger.ALPHA, util.time_to_dist(time))

        trigger[ppt.Alpha.TARGET_ID] = self.target

        trigger[ppt.Alpha.OPACITY] = opacity / 100.0
        trigger[ppt.Alpha.DURATION] = t

        return self

    def _stop_trigger_common(self,
        time: float, target: int | Component, option: STOP_MODE, useControlID: bool):
        target = target.caller if isinstance(target, Component) else target
        validate_params(targets=target)

        trigger = self.create_trigger(obj_id.Trigger.STOP, util.time_to_dist(time))

        trigger[ppt.Stop.TARGET_ID] = target

        trigger[ppt.Stop.MODE] = option # 0=Stop, 1=Pause, 2=Resume
        trigger[ppt.Stop.USE_CONTROL_ID] = useControlID


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

        trigger = self.create_trigger(obj_id.Trigger.COLLISION, util.time_to_dist(time))

        trigger[ppt.Collision.TARGET_ID] = self.target

        trigger[ppt.Collision.BLOCK_A] = blockA
        trigger[ppt.Collision.BLOCK_B] = blockB
        trigger[ppt.Collision.ACTIVATE_GROUP] = activateGroup
        if onExit: trigger[ppt.Collision.EXIT] = True

        return self

    def Count(self, time: float, *, item_id: int, count: int, activateGroup: bool, multiActivate: bool = True):
        validate_params(targets=self.target, item_id=item_id)

        trigger = self.create_trigger(obj_id.Trigger.COUNT, util.time_to_dist(time))

        trigger[ppt.Count.TARGET_ID] = self.target

        trigger[ppt.Count.ITEM_ID] = item_id
        trigger[ppt.Count.COUNT] = count
        trigger[ppt.Count.ACTIVATE_GROUP] = activateGroup
        trigger[ppt.Count.MULTI_ACTIVATE] = multiActivate

        return self

    def Pickup(self, time: float, *, item_id: int, count: int, override: bool):
        """Change an Item ID value by 'count' amount, or set to amount w/ 'override'"""
        validate_params(item_id=item_id)

        if count == 0: raise ValueError("Pickup: Count is 0 (no change)")

        trigger = self.create_trigger(obj_id.Trigger.PICKUP, util.time_to_dist(time))

        trigger[ppt.Pickup.ITEM_ID] = item_id
        trigger[ppt.Pickup.COUNT] = count
        trigger[ppt.Pickup.OVERRIDE] = override
        trigger[ppt.Pickup.MODE] = 0

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

        trigger = self.create_trigger(obj_id.Trigger.PICKUP, util.time_to_dist(time))

        trigger[ppt.Pickup.ITEM_ID] = item_id
        trigger[ppt.Pickup.MODE] = mode
        trigger[ppt.Pickup.MOD] = factor

        return self

    def Timer(self, time: float, *, item: int, 
        start: float, end: float, mod: float = 1, start_paused: bool = False):
        """Activate a time trigger that counts down from 'duration' seconds."""
        validate_params(item_id=item)
        trigger = self.create_trigger(obj_id.Trigger.TIME, util.time_to_dist(time))
        trigger[ppt.Time.TARGET_ID] = self.target

        trigger[ppt.Time.ITEM_ID] = item
        trigger[ppt.Time.START_TIME] = start
        trigger["a473"] = end  # STOP_TIME (missing from obj_prop.Trigger.Time)
        trigger[ppt.Time.MOD] = mod
        trigger[ppt.Time.START_PAUSED] = start_paused

        return self

    def _time_control(self, time: float, item: int, stop: bool):
        validate_params(item_id=item)
        trigger = self.create_trigger(obj_id.Trigger.TIME_CONTROL, util.time_to_dist(time))
        trigger[ppt.TimeControl.ITEM_ID] = item
        trigger[ppt.TimeControl.STOP] = stop
        return self
    
    def TimerStart(self, time: float, *, item: int):
        """Start/Resume a timer."""
        return self._time_control(time, item, stop=False)
    
    def TimerStop(self, time: float, *, item: int):
        """Stop/Pause a timer."""
        return self._time_control(time, item, stop=True)
    
    def TimerOp(self, time: float, *, item: int, mod: float = 1, 
        sign: enum.Item.MathOp = 0): # type: ignore
        """
        Set a Timer ID to a specific value.
        
        Formula: target = target (sign_op) mod

        sign as 'None' removes 'target' from the right side
        """
        validate_params(item_id=item)
        trigger = self.create_trigger(obj_id.Trigger.ITEM_EDIT, util.time_to_dist(time))
        trigger[ppt.ItemEdit.TARGET_ITEM_ID] = item
        
        trigger[ppt.ItemEdit.TARGET_ITEM_TYPE] = enum.Item.ItemType.TIMER.value # type for target
        trigger[ppt.ItemEdit.MOD] = mod
        if sign: trigger[ppt.ItemEdit.ITEM_OP_1] = sign.value # op for 'target = target (sign) (mod)' 
        
        return self

    def TriggerFromObj(self, time: float, trigger: GenericObj | ObjectType):
        """Add a trigger from an object and automatically set it up for the component.
        Takes ownership of 'trigger' (it is modified in place, not copied)."""
        if obj_prop.Y in trigger: del trigger[obj_prop.Y]
        self._register_trigger(trigger, util.time_to_dist(time)) # type: ignore
        return self

    def TriggerFromStr(self, time: float, string: str):
        """Add a trigger from an object string and automatically set it up for the component"""
        return self.TriggerFromObj(time, from_object_string(string))


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

        if any(t[obj_prop.ID] == obj_id.Trigger.SPAWN for t in comp.triggers):
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
                component.Spawn(0, enum.EMPTY_MULTITARGET, True, remap=rb.pairs)
            cls._binary_bases[power] = component

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
                remap_pairs = spawn_trigger.get(ppt.Spawn.REMAPS)
                assert remap_pairs is not None

                remap_callback(remap_pairs, remap)

            remap.pair(enum.EMPTY_MULTITARGET, comp.caller)
            caller.Spawn(time, mt_comp.caller, False,
                remap=remap.pairs, reset_remap=False)
    
    class Template:
        _goto_comp_storage: Component | None = None
        @classmethod
        def goto(cls) -> Component:
            if cls._goto_comp_storage is None:
                cls._goto_comp_storage = Component("PointerCleanup_Goto", unknown_g(), 5)
                with cls._goto_comp_storage.temp_context(target=enum.EMPTY_BULLET):
                    cls._goto_comp_storage \
                        .assert_spawn_order(False) \
                        .GotoGroup(0, enum.EMPTY_TARGET_GROUP)
            return cls._goto_comp_storage


# ===========================================================
#
# PATTERN CLASSES
#
# ===========================================================

# Threshold for using multitarget optimization in CleanPointerCircle
# Batches smaller than this are processed manually to avoid edge case bugs
_MULTITARGET_THRESHOLD = 8

class Pointer:
    def __init__(self, component: Component):
        self._component = component
        self._pc: lib.GuiderCircle | None = None
        self._pc_start_time: float | None = None
    
    @property
    def pc(self) -> lib.GuiderCircle:
        if self._pc is None:
            raise RuntimeError(f"Component '{self._component.name}': No active PointerCircle")
        return self._pc

    def SetPointerCircle(self, time: float, *, location: int, follow: bool, set_north: bool = True):
        """
        Set new pointercircle to allocate pointers into position.
        
        Follow mode:
        - Allocates a new group to the pointers used in this guidercircle.
        - Allows the pointercircle to undergo any transformations (rotate, scale, move)
        """
        if self._pc is not None:
            raise RuntimeError(f"Component '{self._component.name}': PointerCircle already active")

        if follow:
            pc = lib.GuiderCircle(point=0, center=0, all_group=unknown_g())
        else:
            pc = lib.GuiderCircle(point=0, center=0, all_group=0)

        with self._component.temp_context(target=lib.circle1.all):
            self._component.GotoGroup(time - enum.TICK*4, location)
            if set_north:
                self._component.PointToGroup(time - enum.TICK*3, enum.NORTH_GROUP)
        with self._component.temp_context(target=pc.center):
            self._component.GotoGroup(time - enum.TICK*2, lib.circle1.center)

        self._pc = pc
        self._pc_start_time = time
        return self._component

    def CleanPointerCircle(self):
        """Remove active Pointer-based GuiderCircle"""

        c1 = lib.circle1
        pc = self.pc
        time = self._pc_start_time
        assert time is not None

        used_pointers = [(g, c1.groups[i]) for i, g in enumerate(pc.groups) if g != -1]
        if pc.all != 0:
            lib.pointer.register_set(pc.all, [g for g, _ in used_pointers] + [pc.center])

        pair_iter = iter(used_pointers)
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
        goto_comp = Multitarget.Template.goto()
        while remaining > 0:
            batch_size = 64 if remaining > 127 else remaining
            
            if batch_size < _MULTITARGET_THRESHOLD:
                # Manual: small batches to avoid multitarget edge case bugs
                for _ in range(batch_size):
                    pc_pointer, c1_pointer = next(pair_iter)
                    with self._component.temp_context(target=pc_pointer):
                        self._component.GotoGroup(time, c1_pointer)
            else:
                Multitarget.spawn_with_remap(self._component, time, batch_size, goto_comp, remap_goto)

            remaining -= batch_size

        self._pc = None
        self._pc_start_time = None
        return self._component


class BulletAlloc:
    deferred_calls: list[tuple[float, int, Callable[..., Any]]] = []  # (local_time, component_id, func)
    component_offsets: dict[int, float] = {}  # component_id -> stage spawn time
    active: bool = False
    
    @classmethod
    def start(cls):
        if cls.active:
            raise RuntimeError("BulletAlloc.start: BulletAlloc already active")
        cls.active = True

    @classmethod
    def register_spawn_time(cls, component: Component, spawn_time: float):
        """Called by spawn_enemy to tell BulletAlloc when this component fires on stage."""
        if not cls.active:
            raise RuntimeError("BulletAlloc.register_spawn_time: BulletAlloc not active")
        cls.component_offsets[id(component)] = spawn_time

    @classmethod
    def resolve(cls):
        if not cls.active:
            raise RuntimeError("BulletAlloc.resolve: BulletAlloc not active")
        if len(cls.deferred_calls) == 0:
            raise RuntimeError("BulletAlloc.resolve: No deferred calls to resolve")

        cls.deferred_calls.sort(
            key=lambda x: x[0] + cls.component_offsets.get(x[1], 0)
        )
        for _, _, func in cls.deferred_calls:
            func()
        cls.deferred_calls.clear()
        cls.component_offsets.clear()
        cls.active = False

    @classmethod
    def defer(cls, time: float, component: Component, func: Callable[..., None]):
        """Called by pattern methods to queue a bullet allocation."""
        if not cls.active:
            raise RuntimeError("BulletAlloc.defer: BulletAlloc not active")
        cls.deferred_calls.append((time, id(component), func))


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
        
        pc = self._component.pointer.pc

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

        BulletAlloc.defer(time, self._component,
            lambda: Multitarget.spawn_with_remap(self._component, time, numBullets, comp, remap_arc)
        )
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
        type: EASING = 0, rate: float = 1.0):
        """
        Line pattern - builds MoveTowards triggers at different speeds, forming a line.

        Comp requires EMPTY_BULLET and EMPTY_EMITTER.
        Optional: type, rate
        """
        validate_params(positive=[fastestTime, slowestTime], type=type, rate=rate)
        IL = "Instant.Line:"

        util.enforce_component_targets(IL, comp,
            requires={ enum.EMPTY_BULLET, enum.EMPTY_EMITTER },
            excludes={ enum.EMPTY_TARGET_GROUP, enum.EMPTY_MULTITARGET,
                enum.EMPTY1, enum.EMPTY2 }
        )

        if bullet.has_orientation and not comp.has_trigger_properties({ ppt.Rotate.AIM_MODE: Any }):
            warn(f"{IL} Bullet has orientation enabled, but component has no PointToGroup trigger."
                 f"Bullets may not face the correct direction.")

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
        
        def deferred():
            Multitarget.spawn_with_remap(self._component, time, numBullets, comp, remap_line)
    
            step = (slowestTime - fastestTime) / (numBullets - 1)
            for i, bullet_group in enumerate(bullet_groups):
                travel_time = fastestTime + step * i
                with self._component.temp_context(target=bullet_group):
                    self._component.MoveTowards(
                        time + enum.TICK*2, targetDir,
                        t=travel_time, dist=dist, type=type, rate=rate
                    )

        BulletAlloc.defer(time, self._component, deferred)

        return self._component

    # More pattern methods will be added here


class TimedPatterns:
    def __init__(self, component: Component):
        self._component = component

    def RadialWave(self, time: float, comp: Component, bullet: lib.BulletPool, *,
        waves: int, interval: float, numBullets: int | None = None, spacing: int | None = None, centerAt: float = 0):
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
        numBullets: int, spacing: float, t: float, dist: int, type: EASING = 0, rate: float = 1.0):
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

        if bullet.has_orientation and not comp.has_trigger_properties({ppt.Rotate.AIM_MODE:Any}):
            warn(f"{TL} Bullet has orientation enabled, but component has no PointToGroup triggers. Bullets may not face the correct direction.")

        if numBullets < 2:
            raise ValueError(f"{TL} numBullets must be at least 2. Got: {numBullets}")
        if spacing < 0:
            raise ValueError(f"{TL} spacing must be non-negative. Got: {spacing}")
        if t < 0:
            raise ValueError(f"{TL} t must be non-negative. Got: {t}")

        for i in range(numBullets):
            b, _ = bullet.next()
            self._component.Spawn(
                time + (i * spacing), comp.caller, True, remap={ enum.EMPTY_BULLET: b })
            with self._component.temp_context(target=b):
                self._component.MoveTowards(
                    time + (i * spacing), targetDir, t=t, dist=dist, type=type, rate=rate
                )
        return self._component
    
    def BezierMove(self, time: float, curve_label: CurveType, *, 
        dx: float, dy: float, t: float, generate_preview: bool = False):
        """
        Apply smooth Bezier curved movement using pre-registered curves.
        
        Movement is RELATIVE to the object's position.
        
        Args:
            curve_label: Curve name string or CurveType enum
            generate_preview: If True, generates an animated GIF preview of the movement
        
        Setting up a curve type:
            # Register curves once (at startup or in separate script)
            from touhou_scs.movements import register_bezier_curve
            register_bezier_curve("gentle_arc",
                p0=(0, 0), p1=(0.2, 0.6), p2=(0.8, 1.2), p3=(1, 1),
                quality="medium")
        """
        validate_params(non_negative=[t, time])
        enforce_solid_groups(self._component.target)
        return apply_bezier_movement(
            self._component, time, curve_label,
            dx, dy, t, generate_preview
        )

    # More pattern methods will be added here
