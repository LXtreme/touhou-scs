"""Additional Component helpers, exposed through the builder properties."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gmdbuilder import ObjectType, new_obj, obj_id, obj_prop
from gmdbuilder.classes import ItemCompare

from touhou_scs import enums as enum
from touhou_scs.types import GenericObj
from touhou_scs.utils import unknown_g

if TYPE_CHECKING:
    from touhou_scs.component import Component

ppt = obj_prop.Trigger


class RandomPatterns:
    """Synchronous numeric RNG and relative rotation using scratch TIMER 38.

    All calls overwrite timer 38. Numeric work finishes through direct Random /
    Item Compare callbacks; Rotate still applies during GD's movement update.
    No remaps, parallel calls across ticks, automatic aiming, or firing loops.
    """

    TIMER_ID = 38
    BITS = 16
    _logic: Component | None = None
    _rotation_entries: dict[tuple[int, int], int] = {}

    def __init__(self, component: Component):
        self._component = component
        self._range_group = 0

    @classmethod
    def _storage(cls) -> Component:
        from touhou_scs import lib
        from touhou_scs.component import Component

        # A new build/test registry needs fresh triggers and group IDs.
        if cls._logic is None or cls._logic not in lib.all_components:
            cls._logic = Component("Random: RNG and binary rotates", unknown_g(), 6) \
                .assert_spawn_order(True)
            cls._rotation_entries = {}
            entry = cls._logic.caller
            cls._timer_op(entry, 0)
            for bit in range(cls.BITS):
                branch = unknown_g()
                trigger = new_obj(obj_id.Trigger.RANDOM)
                trigger[ppt.Random.TRUE_ID] = branch
                trigger[ppt.Random.CHANCE] = 50
                cls._emit(entry, trigger)
                cls._timer_op(branch, 2 ** bit, enum.Item.MathOp.ADD)
            cls._timer_op(entry, 2 ** cls.BITS, enum.Item.MathOp.DIVIDE)
        return cls._logic

    @classmethod
    def _emit(cls, groups: int | set[int], trigger: GenericObj | ObjectType):
        logic = cls._logic
        assert logic is not None
        # Explicit X order survives export. Callback groups are NOT Spawn Ordered,
        # so these distances enforce order without introducing internal delays.
        time = len(logic.triggers) * 2 / enum.PLR_SPEED
        with logic.temp_context(groups=groups, include_caller=False):
            logic.TriggerFromObj(time, trigger)

    @classmethod
    def _timer_op(cls, groups: int | set[int], value: float,
        operation: enum.Item.MathOp = enum.Item.MathOp.ASSIGN):
        logic = cls._logic
        assert logic is not None
        with logic.temp_context(groups=groups, include_caller=False):
            logic.TimerOp(len(logic.triggers) * 2 / enum.PLR_SPEED,
                item=cls.TIMER_ID, mod=value, sign=operation)

    @classmethod
    def _compare(cls, groups: int | set[int], expression: str, true_id: int, false_id: int = 0):
        compare = ItemCompare().set_from_string(expression)
        compare.true_id = true_id
        compare.false_id = false_id
        compare.tolerance = 0
        cls._emit(groups, compare.obj)

    @staticmethod
    def _call(component: Component, time: float, group: int):
        # An always-true comparison is the synchronous entry point, not a Spawn.
        compare = ItemCompare().set_from_string("i38 == i38")
        compare.true_id = group
        compare.tolerance = 0
        component.TriggerFromObj(time, compare.obj)

    def rand_range(self, time: float, minimum: float, maximum: float) -> Component:
        """Write a fresh 16-bit uniform sample in [minimum, maximum) to TIMER 38.

        Equal bounds produce that constant. Returns the parent Component for
        chaining. Uses one scratch timer shared by all synchronous calls.
        """
        if not (math.isfinite(minimum) and math.isfinite(maximum)):
            raise ValueError("rand_range: bounds must be finite")
        if minimum > maximum:
            raise ValueError("rand_range: minimum must not exceed maximum")
        if max(abs(minimum), abs(maximum), maximum - minimum) > 9_999_999:
            raise ValueError("rand_range: bounds and span must fit GD's timer limit")

        logic = self._storage()
        entry = unknown_g()
        self._range_group = entry
        if minimum == maximum:
            self._timer_op(entry, minimum)
        else:
            self._compare(entry, "i38 == i38", logic.caller)
            self._timer_op(entry, maximum - minimum, enum.Item.MathOp.MULTIPLY)
            self._timer_op(entry, minimum, enum.Item.MathOp.ADD)
        self._call(self._component, time, entry)
        return self._component

    @classmethod
    def _rotation_entry(cls, target: int, center: int) -> int:
        cls._storage()
        key = (target, center)
        if key in cls._rotation_entries:
            return cls._rotation_entries[key]

        entry, negative, check = unknown_g(), unknown_g(), unknown_g()
        cls._rotation_entries[key] = entry
        cls._compare(entry, "t38 < 0", negative, check)
        # One full turn makes negative angles nonnegative without another timer
        # or a separate bank of negative Rotate triggers.
        cls._timer_op(negative, 360, enum.Item.MathOp.ADD)
        previous_take = None
        for weight in (256, 128, 64, 32, 16, 8, 4, 2, 1):
            take = unknown_g()
            next_check = unknown_g() if weight > 1 else 0
            groups = {check}
            if previous_take is not None:
                groups.add(previous_take)
            else:
                groups.add(negative)
            cls._compare(groups, f"t38 >= {weight}", take, next_check)
            if next_check:
                cls._timer_op(take, weight, enum.Item.MathOp.SUBTRACT)
            rotate = new_obj(obj_id.Trigger.ROTATE)
            rotate[ppt.Rotate.TARGET_ID] = target
            rotate[ppt.Rotate.CENTER_ID] = center
            rotate[ppt.Rotate.DEGREES] = weight
            rotate[ppt.Rotate.DURATION] = 0
            cls._emit(take, rotate)
            check, previous_take = next_check, take
        return entry

    def rotate(self, time: float, *, spread: float, offset: float = 0, center: int | None = None) -> Component:
        """Add random(offset - spread, offset + spread) to the current angle.

        Uses the current target context and a reusable Rotate bank per target /
        center pair. Bounds must lie within [-360, 360]. The sampled angle is
        rounded down to whole-degree steps (error < 1 degree); timer 38 is
        consumed by the converter. Normalize/aim explicitly before calling.
        """
        from touhou_scs.component import enforce_solid_groups, validate_params

        target = self._component.target
        if center is None:
            center = target
        validate_params(targets=[target, center], non_negative=spread)
        minimum, maximum = offset - spread, offset + spread
        if not (math.isfinite(minimum) and math.isfinite(maximum)):
            raise ValueError("random.rotate: spread and offset must be finite")
        if minimum < -360 or maximum > 360:
            raise ValueError("random.rotate: angle bounds must lie within [-360, 360]")
        enforce_solid_groups(target, center)
        self.rand_range(time, minimum, maximum)
        converter = self._rotation_entry(target, center)
        self._compare(self._range_group, "i38 == i38", converter)
        return self._component


