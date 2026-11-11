"""Additional Component helpers, exposed through the builder properties."""

from __future__ import annotations

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

    def __init__(self, component: Component):
        self._component = component
        self._range_group = 0

    @classmethod
    def _get_bit_bank(cls, bits: int) -> tuple[Component, int]:
        """Get shared storage and the bank caller, creating them as needed."""
        from touhou_scs.component import Component

        if cls._rng_storage is None:
            cls._rng_storage = Component("Random timers", unknown_g()).assert_spawn_order(False)
        storage = cls._rng_storage
        if bits in cls._rand_bits:
            return cls._rng_storage, cls._rand_bits[bits]
        
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
        
        return storage, cls._rand_bits[bits]

    def rand_timer(self, time: float, minimum: float, maximum: float, *, 
        t: int, bits: int = BITS):
        """
        Write a bits-bit random sample in [minimum, maximum) to timer 't'.

        Entries are shared by (minimum, maximum, bits), with a remapped output timer.
        The Spawn entry may add latency; arithmetic uses synchronous callbacks.
        """
        from touhou_scs.component import validate_params

        validate_params(item_id=t)
        # Larger raw samples exceed GD's timer assignment limit (9,999,999).
        if not 1 <= bits <= 23:
            raise ValueError("rand_timer: bits must be between 1 and 23")
        if not minimum < maximum:
            raise ValueError("rand_timer: minimum must be less than maximum")

        key = (minimum, maximum, bits)
        if key not in self._range_entries:
            storage, bank = self._get_bit_bank(bits)
            entry = unknown_g()
            sample = ItemCompare().set_from_string("0 == 0")
            sample.true_id = bank

            # Normalize separately: Item Edit modifiers retain only three decimals.
            with storage.temp_context(groups=entry, include_caller=False):
                storage.TimerOp(0, item=enum.EMPTY1, mod=0)
                storage.TriggerFromObj(1, sample.obj)
                storage.TimerOp(2, item=enum.EMPTY1, mod=2 ** bits, sign=enum.Item.MathOp.DIVIDE)
                storage.TimerOp(3, item=enum.EMPTY1, mod=maximum - minimum,
                    sign=enum.Item.MathOp.MULTIPLY)
                if minimum != 0:
                    storage.TimerOp(4, item=enum.EMPTY1, mod=minimum, sign=enum.Item.MathOp.ADD)
            self._range_entries[key] = entry

        entry = self._range_entries[key]
        self._component.Spawn(time, entry, False, remap={enum.EMPTY1: t})
        return self._component

    @classmethod
    def _rotation_entry(cls, target: int, center: int) -> int:
        logic = cls._storage()
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

            # Explicit X order survives export. Callback groups are NOT Spawn Ordered,
            # so these distances enforce order without introducing internal delays.
            time = len(logic.triggers) * 2 / enum.PLR_SPEED
            with logic.temp_context(groups=take, target=target, include_caller=False):
                logic.Rotate(time, angle=weight, center=center, t=0)

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
        enforce_solid_groups(target, center)

        minimum = offset - spread
        maximum = offset + spread
        if minimum < -360 or maximum > 360:
            raise ValueError("random.rotate: angle bounds must lie within [-360, 360]")
        
        self.rand_range(time, minimum, maximum)
        converter = self._rotation_entry(target, center)
        self._compare(self._range_group, "i38 == i38", converter)

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
