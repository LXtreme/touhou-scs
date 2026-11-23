"""Additional Component helpers, exposed through the builder properties."""

from __future__ import annotations

from decimal import Decimal
from itertools import cycle
from typing import Any, TYPE_CHECKING

from gmdbuilder import new_obj, obj_id, obj_prop
from gmdbuilder.classes import ItemCompare
from gmdbuilder.object_types import EASING

from touhou_scs import enums as enum, utils as util
from touhou_scs.movements import CurveType, apply_bezier_movement
from touhou_scs.utils import unknown_g, warn

if TYPE_CHECKING:
    from touhou_scs import lib
    from touhou_scs.component import Component

ppt = obj_prop.Trigger


class RandomPatterns:
    BITS = 12
    _rand_bits: dict[int, int] = {} # bit count -> shared bank caller
    _range_entries: dict[tuple[float, float, int], int] = {}
    _rng_storage: Component | None = None
    _rotate_entries: dict[tuple[float, float, int], int] = {}
    _rotation_converter: int | None = None

    def __init__(self, component: Component):
        self._component = component
        self._rotate_timers = cycle(range(60, 80)) # scratch timer IDs

    @classmethod
    def _get_bit_bank(cls, bits: int) -> tuple[Component, int]:
        """Get shared storage and the bank caller, creating them as needed."""
        from touhou_scs.component import Component

        if cls._rng_storage is None:
            cls._rng_storage = Component("Random: Shared Triggers", 0).assert_spawn_order(False)
        storage = cls._rng_storage
        if bits in cls._rand_bits:
            return storage, cls._rand_bits[bits]
        
        caller = unknown_g()
        with storage.temp_context(groups=caller, include_caller=False):
            for bit in range(bits):
                take = unknown_g()
                flip = new_obj(obj_id.Trigger.RANDOM)
                flip[ppt.Random.CHANCE] = 50
                flip[ppt.Random.TRUE_ID] = take
                storage.TriggerFromObj(bit, flip)
                with storage.temp_context(groups=take, include_caller=False):
                    storage.TimerOp(0, item=enum.EMPTY1, mod=2 ** bit, sign=enum.Item.MathOp.ADD)
        cls._rand_bits[bits] = caller
        
        return storage, caller

    @classmethod
    def _get_range_entry(cls, minimum: float, maximum: float, bits: int) -> tuple[Component, int]:
        # Larger raw samples exceed GD's timer assignment limit (9,999,999).
        if not 1 <= bits <= 23:
            raise ValueError("rand_timer: bits must be between 1 and 23")
        if not minimum < maximum:
            raise ValueError("rand_timer: minimum must be less than maximum")

        storage, bank = cls._get_bit_bank(bits)
        key = (minimum, maximum, bits)
        if key in cls._range_entries:
            return storage, cls._range_entries[key]

        entry = unknown_g()
        # Normalize separately: Item Edit modifiers retain only three decimals.
        with storage.temp_context(groups=entry, include_caller=False):
            storage.TimerOp(0, item=enum.EMPTY1, mod=0)
            cls._compare(storage, 1, "0 == 0", bank)
            storage.TimerOp(2, item=enum.EMPTY1, mod=2 ** bits, sign=enum.Item.MathOp.DIVIDE)
            storage.TimerOp(3, item=enum.EMPTY1, mod=maximum - minimum, sign=enum.Item.MathOp.MULTIPLY)
            if minimum != 0:
                storage.TimerOp(4, item=enum.EMPTY1, mod=minimum, sign=enum.Item.MathOp.ADD)
        cls._range_entries[key] = entry
        return storage, entry

    @staticmethod
    def _compare(storage: Component, time: float, expression: str, on_true: int) -> None:
        compare = ItemCompare().set_from_string(expression)
        compare.true_id = on_true
        compare.tolerance = 0
        storage.TriggerFromObj(time, compare.obj)

    def rand_timer(self, time: float, minimum: float, maximum: float, *,
        t: int | None = None, bits: int = BITS) -> Component:
        """Write a sample in [minimum, maximum) to timer t.

        Omit t to cycle through scratch timers 60-79 during generation.
        Use an explicit timer when the value must outlive scratch reuse.
        """
        from touhou_scs.component import validate_params

        validate_params(item_id=t)
        _, entry = self._get_range_entry(minimum, maximum, bits)
        if t is None:
            t = next(self._rotate_timers)
        self._component.Spawn(time, entry, False, remap={enum.EMPTY1: t})
        return self._component

    @classmethod
    def _get_rotation_converter(cls, storage: Component) -> int:
        if cls._rotation_converter is not None:
            return cls._rotation_converter

        converter, wrap = unknown_g(), unknown_g()
        timer = enum.EMPTY1
        units_per_degree = 10
        full_turn = 360 * units_per_degree
        with storage.temp_context(groups=wrap, include_caller=False):
            storage.TimerOp(0, item=timer, mod=full_turn, sign=enum.Item.MathOp.ADD)

        # Use integer weights in tenths. Wrap negatives to avoid a second bank.
        with storage.temp_context(groups=converter, include_caller=False):
            storage.TimerOp(0, item=timer, mod=units_per_degree, sign=enum.Item.MathOp.MULTIPLY)
            cls._compare(storage, 1, f"t{timer} < 0", wrap)
            for bit in range(full_turn.bit_length() - 1, -1, -1):
                weight = 2 ** bit
                take = unknown_g()
                cls._compare(storage, full_turn.bit_length() + 1 - bit, f"t{timer} >= {weight}", take)
                with storage.temp_context(groups=take, target=enum.EMPTY_TARGET_GROUP, include_caller=False):
                    storage.TimerOp(0, item=timer, mod=weight, sign=enum.Item.MathOp.SUBTRACT)
                    storage.Rotate(1, angle=weight / units_per_degree, center=enum.EMPTY2)

        cls._rotation_converter = converter
        return converter

    def rotate(self, time: float, *, spread: float, offset: float = 0, center: int | None = None,
        t: int | None = None, bits: int = BITS) -> Component:
        """Add a random angle in [offset - spread, offset + spread).

        Bounds must lie within [-360, 360]. Sampled angles round down to tenths
        of a degree. For nonzero spread, the lower bound and range width must
        be multiples of 0.001 degrees so Item Edit can store them exactly.
        The converter consumes t or a scratch timer if omitted.
        Aim explicitly before calling when you need an absolute reference.

        Tick Interval Reliability [limits may need more testing]:
        4 ticks between rerolls, Rotate Follow mode 2 tick after initial time.
        """
        from touhou_scs.component import enforce_solid_groups, validate_params

        target = self._component.target
        if center is None:
            center = target
        validate_params(targets=[target, center], non_negative=spread, item_id=t)

        minimum = offset - spread
        maximum = offset + spread
        if not -360 <= minimum <= maximum <= 360:
            raise ValueError("random.rotate: angle bounds must lie within [-360, 360]")

        # Check the two stored modifiers before binary float subtraction
        # can make an exact decimal input appear to exceed GD's precision.
        decimal_spread = Decimal(str(spread))
        lower_bound = Decimal(str(offset)) - decimal_spread
        width = 2 * decimal_spread
        precision = Decimal("0.001")
        if lower_bound % precision or width % precision:
            raise ValueError(
                "random.rotate: the lower bound and range width must be multiples of "
                "0.001 degrees (Item Edit modifier precision)"
            )

        enforce_solid_groups(target, center)

        key = (minimum, maximum, bits)
        if key not in self._rotate_entries:
            storage, rng_entry = self._get_range_entry(minimum, maximum, bits)
            converter = self._get_rotation_converter(storage)
            entry = unknown_g()
            with storage.temp_context(groups=entry, include_caller=False):
                self._compare(storage, 0, "0 == 0", rng_entry)
                self._compare(storage, 1, "0 == 0", converter)
            self._rotate_entries[key] = entry

        t = t or next(self._rotate_timers)
        self._component.Spawn(time, self._rotate_entries[key], False, 
            remap={enum.EMPTY1: t, enum.EMPTY_TARGET_GROUP: target, enum.EMPTY2: center})
        return self._component


class InstantPatterns:
    def __init__(self, component: Component):
        self._component = component


    def Arc(self, time: float, comp: Component, bullet: lib.BulletPool, *,
        numBullets: int, angle: float, centerAt: float = 0, _radialBypass: bool = False) -> Component:
        """
        Arc pattern - partial circle of bullets

        Component must use EMPTY_BULLET and EMPTY_TARGET_GROUP. \n
        Optional: centerAt
        """
        from touhou_scs.component import BulletAlloc, Multitarget

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
        numBullets: int | None = None, spacing: int | None = None, centerAt: float = 0) -> Component:
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
        type: EASING = 0, rate: float = 1.0) -> Component:
        """
        Line pattern - builds MoveTowards triggers at different speeds, forming a line.

        Comp requires EMPTY_BULLET and EMPTY_EMITTER.
        Optional: type, rate
        """
        from touhou_scs.component import BulletAlloc, Multitarget, validate_params

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
        waves: int, interval: float, numBullets: int | None = None, spacing: int | None = None, centerAt: float = 0) -> Component:
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
        numBullets: int, spacing: float, t: float, dist: int, type: EASING = 0, rate: float = 1.0) -> Component:
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
        dx: float, dy: float, t: float, generate_preview: bool = False) -> Component:
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
        from touhou_scs.component import enforce_solid_groups, validate_params

        validate_params(non_negative=[t, time])
        enforce_solid_groups(self._component.target)
        return apply_bezier_movement(
            self._component, time, curve_label,
            dx, dy, t, generate_preview
        )

    # More pattern methods will be added here
