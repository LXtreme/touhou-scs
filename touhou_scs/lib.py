"""
Touhou SCS - Library Module

Core infrastructure: Spell system, GuiderCircles, BulletPools, and export functionality.
Module-level storage for components and spells for automatic registration.
"""

from itertools import cycle
import random
import time
import colorsys
from typing import Any, ClassVar, Self

from touhou_scs import enums as enum
from touhou_scs import utils as util
from touhou_scs.component import Component, BulletAlloc
from touhou_scs.utils import unknown_g, warn
from touhou_scs.types import ComponentProtocol, SpellProtocol, GenericObj, TriggerArea
from dataclasses import dataclass

from gmdbuilder import Level, from_object_string, obj_id, obj_prop

ppt = obj_prop.Trigger

all_spells: list[SpellProtocol] = []
all_components: list[ComponentProtocol] = []

# unfortunately has to be up here to avoid circular imports
solid_groups_to_enforce: set[int] = set()
registered_bullet_pools: list["BulletPool"] = []  # populated by BulletPool.__init__; used for validation exemptions
registered_enemy_pools: list["EnemyPool"] = []  # populated by EnemyPool.__init__; used by add_enemy_collisions
def _validate_solid_groups(*specific_groups: int):
    """Ensures groups marked as solid are not component or trigger groups."""
    groups = specific_groups if specific_groups else solid_groups_to_enforce

    for g in groups:
        for c in all_components:
            # Check if it's a non-solid group (i.e. if it belongs to a comp/trigger)
            if g in c.groups:
                raise ValueError(f"Group {g} is a component caller in '{c.name}', not a solid group")
            
            if any(pool.min_group <= g <= pool.max_group for pool in registered_bullet_pools):
                continue
            if any(pool.min_group <= g <= pool.max_group for pool in registered_enemy_pools):
                continue
            
            for t in c.triggers:
                if t[obj_prop.ID] == enum.POINTER_OBJ_ID: continue
                
                if g in t.get(obj_prop.GROUPS, []):
                    raise ValueError(f"Group {g} is a trigger group ({t}) in '{c.name}', not a solid group")


_start_time = time.time()

DEFAULT_TRIGGER_AREA: TriggerArea = {
    "min_x": 1350,
    "min_y": 1300,
    "max_x": 20000,
    "max_y": 4500
}

class Spell:
    def __init__(self, spell_name: str, caller_group: int):
        self.spell_name: str = spell_name
        self.caller_group: int = caller_group
        self.components: list[ComponentProtocol] = []
        all_spells.append(self)

    def add_component(self, component: ComponentProtocol) -> Self:
        self.components.append(component)
        return self


# ============================================================================
# USEFUL/NEEDED UTILS
# ============================================================================

@dataclass(slots=True)
class HSB:
    h: float
    s: float
    b: float

def rgb(r: float, g: float, b: float) -> HSB:
    """Convert RGB into HSB adjustments for converting red into desired color."""
    # Normalize RGB to [0, 1]
    r, g, b = r / 255, g / 255, b / 255
    h, s, b = colorsys.rgb_to_hsv(r, g, b)

    base_h, base_s, base_b = 0.0, 1.0, 1.0 # Base red in HSV is (0°, 1, 1)

    hue_offset = (h * 360.0) - (base_h * 360.0)

    # Wrap hue to -180..180
    if hue_offset > 180: hue_offset -= 360
    elif hue_offset < -180: hue_offset += 360

    sat_offset = s - base_s
    bright_offset = b - base_b

    return HSB(hue_offset, sat_offset, bright_offset)

# ============================================================================
# IN-LEVEL GROUP ASSIGNMENTS
# (i.e. real objects like bullets, guidercircle, emitters)
# ============================================================================

class GuiderCircle:
    """
    Circle of 1080 pointer objects for angle-based aiming (1/3 degree precision):
    - Circle1: Pre-allocated.
    - Lazy (point == -1): Initializes with zeros, allocates on-demand via angle_to_groups
    """
    PRECISION = 1080

    def __init__(self, center: int, point: int, all_group: int):
        self.all = all_group
        self.center = center if center != 0 else pointer.next()
        self.point = point  # direction that the guidercircle points

        if point != 0: # Circle1
            self.groups = [point + i for i in range(self.PRECISION)]
        else:
            self.groups = [-1] * self.PRECISION
            self.groups[0] = pointer.next()  # Allocate first pointer

    def angle_to_groups(self, startAngle: float, endAngle: float, numPoints: int, closed_circle: bool = False):
        """
        Convert angles to guidercircle groups, snapping to grid.

        Lazily allocates pointers from the pointer pool if not already allocated.
        Returns list of pointer groups corresponding to the requested angles.
        """
        if numPoints < 2:
            raise ValueError("GuiderCircle.angle_to_indices: numPoints must be at least 2")

        points_per_degree = self.PRECISION / 360
        arc_length = endAngle - startAngle

        if closed_circle:
            spacing = arc_length / numPoints
        else:
            spacing = arc_length / (numPoints - 1)

        original_angles = [startAngle + (spacing * i) for i in range(numPoints)]

        groups: list[int] = []
        for orig_angle in original_angles:
            index = round(orig_angle * points_per_degree) % self.PRECISION

            if self.groups[index] == -1: # Lazy allocation
                self.groups[index] = pointer.next()

            groups.append(self.groups[index])

        return groups

circle1 = GuiderCircle(center=6181, point=5101, all_group=6181)

class GuiderLine:
    """Odd number of pointer objects aligned in a straight line"""

    def __init__(self, numPointers: int, groups: list[int]):
        groups.sort()

        if len(groups) % 2 == 0:
            raise ValueError("GuiderLine: Initialized with even number groups!")
        if len(groups) == len(set(groups)):
            raise ValueError("GuiderLine: Initialized with duplicate groups!")

        self.groups = groups
        self.numPointers = numPointers
        self.center = groups[int((len(groups) + 1) / 2)]
        self.end1 = groups[0]
        self.end2 = groups[-1]


class BulletPool:
    """Bullet pool with group range and cycler for sequential allocation."""

    def __init__(self, min_group: int, max_group: int, has_orientation: bool):
        """Inclusive of both min_group and max_group."""
        self.min_group = min_group
        self.max_group = max_group
        self.has_orientation = has_orientation
        self.current = max_group
        registered_bullet_pools.append(self)
        pool_size = max_group - min_group + 1
        # Reserve both the hitbox range and the collision-group range
        util.unknown_g.reserve_range(min_group, max_group + pool_size)

    def next(self) -> tuple[int, int]:
        """Returns: (bullet_group, collision_group)"""
        self.current += 1
        if self.current > self.max_group:
            self.current = self.min_group

        collision = self.current + (self.max_group - self.min_group + 1)
        return self.current, collision


bullet1 = BulletPool(501, 1000, True)
bullet2 = BulletPool(1501, 2200, True)
bullet3 = BulletPool(2901, 3600, False)
bullet4 = BulletPool(4301, 4700, False)

p_pickup = BulletPool(301, 370, True)
b_pickup = BulletPool(371, 380, True)
score_pickup = BulletPool(381, 490, True)

PICKUP_SPAWN_TIME = 0.2
"""Total time window (seconds) over which dropped pickups are spread on enemy death."""


class pointer:
    obj_count = 250
    pointer_comp: ClassVar[Component] = Component("Pointers", 0, 11).assert_spawn_order(False)
    pointers: ClassVar[list[int]] = [unknown_g() for _ in range(obj_count)]
    _pointer_iter: ClassVar = cycle(pointers)
    
    group_registry: ClassVar[dict[int, set[int]]] = { g: {enum.DEBUG_UI_GROUP, g} for g in pointers }
    
    @classmethod
    def next(cls) -> int:
        if cls.export_mappings.has_been_called:
            raise RuntimeError("Pointer.next() cannot be called after export_mappings()")
        return next(cls._pointer_iter)
    
    @classmethod
    def register_set(cls, new_group: int, pointers: list[int]):
        """Register a set of pointers to share a new group mapping."""
        if cls.export_mappings.has_been_called:
            raise RuntimeError("Pointer.register_set cannot be called after export_mappings()")
        for p in pointers:
            cls.group_registry[p].add(new_group)
    
    @classmethod
    @util.calltracker
    def export_mappings(cls):
        """Call at export time to generate pointer objects with group mappings."""
        if cls.export_mappings.has_been_called:
            raise RuntimeError("Pointer.export_mappings has already been called")
        for group_list in cls.group_registry.values():
            # old pointer obj str: 1,3802,2,2625,3,915,20,2,57,7000.33,64,1,67,1,155,14905,25,9,24,11,128,0.25,129,0.25;
            cls.pointer_comp.triggers.append({ #type: ignore
                obj_prop.ID: enum.POINTER_OBJ_ID,
                obj_prop.X: 0.0, obj_prop.Y: 0.0,
                obj_prop.GROUPS: group_list,
                obj_prop.EDITOR_L1: 11,
                obj_prop.OLD_SCALE: 0.2, #type: ignore
                obj_prop.Z_LAYER: 9, # Z layer
                obj_prop.DONT_FADE: True, # dont fade
                obj_prop.DONT_ENTER: True, # dont enter
            })

reimuA_level1 = BulletPool(6300, 6339, True)
reimuA_homing_shots = BulletPool(6230, 6299, True)
reimuA_bomb_balls = BulletPool(6200, 6229, True)
reimuA_needles = BulletPool(6380, 6431, True)


def get_all_components() -> list[ComponentProtocol]: return all_components

class Stage:
    stage1 = Component("Stage1", unknown_g(), 9).assert_spawn_order(True)
    # stage2 = Component("Stage2", unknown_g(), 9).assert_spawn_order(True)
    # stage3 = Component("Stage3", unknown_g(), 9).assert_spawn_order(True)
    # stage4 = Component("Stage4", unknown_g(), 9).assert_spawn_order(True)
    # stage5 = Component("Stage5", unknown_g(), 9).assert_spawn_order(True)
    # stage6 = Component("Stage6", unknown_g(), 9).assert_spawn_order(True)


class EnemyPool:

    def __init__(self, min_group: int, max_group: int, despawn_setup: Component, *, num_groups_per: int = 1):
        if (max_group - min_group + 1) % num_groups_per != 0:
            raise ValueError(f"Pool size must be divisible by num_groups_per={num_groups_per}")

        self.min_group = min_group
        self.max_group = max_group
        self.num_groups_per = num_groups_per
        self._despawn_setup = despawn_setup

        self._current = min_group
        self.__firstcall = True
        # Keys on the first (main) group of each enemy slot
        self._off_switches = {
            g: unknown_g()
            for g in range(min_group, max_group + 1, num_groups_per)
        }
        registered_enemy_pools.append(self)

    def next(self) -> tuple[int, ...]:
        """Cycle to next enemy slot in pool. Returns a tuple of all groups for that slot."""
        if self.__firstcall:
            self.__firstcall = False
        else:
            self._current += self.num_groups_per
            if self._current > self.max_group:
                self._current = self.min_group

        return tuple(range(self._current, self._current + self.num_groups_per))

    @staticmethod
    def _new_drop_comp(drops: list[tuple["BulletPool", int]]) -> Component:
        """
        Create a new drop component with freshly allocated pickup groups.

        The component uses EMPTY_TARGET_GROUP as the enemy position, which gets
        remapped to the real enemy group at spawn time via despawnSetup's remap.
        Each pickup is toggled on and instantly snapped to the enemy's position.
        """
        # Build a human-readable name from pool ranges
        name_parts = [f"{pool.min_group}-{pool.max_group}x{count}" for pool, count in drops]
        comp = Component(f"Drop [{', '.join(name_parts)}]", unknown_g(), editorLayer=7)
        comp.assert_spawn_order(True)

        total = sum(count for _, count in drops)
        step = PICKUP_SPAWN_TIME / max(total - 1, 1)
        i = 0
        for pool, count in drops:
            for _ in range(count):
                pickup_group, _ = pool.next()
                t = i * step
                with comp.temp_context(target=pickup_group):
                    comp.GotoGroup(t, enum.EMPTY_TARGET_GROUP)
                    comp.Toggle(t, True)
                    comp.Spawn(t, enum.PICKUP_RANDOM_MOVE, False, remap={10: pickup_group})
                i += 1

        return comp

    def spawn_enemy(self, stage: Component, time: float, attack: Component, hp: int, enemy_groups: tuple[int, ...],
                    drops: list[tuple["BulletPool", int]] | None = None):
        """Spawn an enemy attack with HP/death handling.

        drops: optional list of (BulletPool, count) pairs specifying what pickups to drop on death.
               e.g. [(lib.score_pickup, 3), (lib.p_pickup, 1)]
               Identical configs reuse the same component — no extra groups per repeated call.
        """
        enemy_group = enemy_groups[0]  # main group used for HP, position, despawn
        if not (self.min_group <= enemy_group <= self.max_group):
            raise ValueError(
                f"spawn_enemy: enemy_group {enemy_group} is not in pool range "
                f"{self.min_group}-{self.max_group}"
            )

        self.time = time

        if BulletAlloc.active:
            BulletAlloc.register_spawn_time(attack, time)

        util.enforce_component_targets("Spawn Enemy", attack,
            excludes={ enum.EMPTY_BULLET, enum.EMPTY1, enum.EMPTY2, enum.EMPTY_EMITTER, enum.EMPTY_MULTITARGET, enum.EMPTY_TARGET_GROUP })

        off_switch = self._off_switches[enemy_group]

        drop_caller = self._new_drop_comp(drops).caller if drops else enum.EMPTY_MULTITARGET

        stage.TimerOp(time - enum.TICK*2, item=enemy_group, mod=hp)
        with stage.temp_context(target=enemy_group):
            stage.Toggle(time - enum.TICK, True)

        with stage.temp_context(groups=off_switch):
            stage.Spawn(time, attack.caller, True)

        ENEMY_HP_CHECK_LOOP = 150
        stage.Spawn(time, ENEMY_HP_CHECK_LOOP, False,
            remap={enum.EMPTY_TARGET_GROUP: enemy_group, enum.EMPTY_BULLET: self._despawn_setup.caller, enum.EMPTY1: off_switch, enum.EMPTY2: drop_caller}
        )


# less annoying way instead of making 'despawner' have spawn order (uses spawn delay instead)
toggler = (Component("Toggler", unknown_g(), 7)
    .assert_spawn_order(False)
    .set_context(target=enum.EMPTY_TARGET_GROUP)
        .Toggle(0, False)
    .clear_context()
)

ret = from_object_string("1,3661,2,0,20,4,62,1,87,1,36,1,51,136,71,26,566,1,568,1;")
ret["a57"] = {toggler.caller}
toggler.triggers.append(ret) # type: ignore

despawner = (Component("Despawner", unknown_g(), 7)
    .assert_spawn_order(False)
    .set_context(target=enum.EMPTY_TARGET_GROUP)
        .Alpha(0, t=0.5, opacity=0)
        .Pulse(0, HSB(0, 0, -20), fadeIn=0.1, t=0.2, fadeOut=0.6, exclusive=True)
        .Scale(0, factor=0.1, t=0.3, hold=1)
    # .set_context(target=130)
    #     .GotoGroup(0, enum.NORTH_GROUP)
    .clear_context()
    .Stop(0, target=enum.EMPTY1)
    # specific sfx 
    .TriggerFromStr(0, "1,3602,2,225,3,-165,155,1,62,1,87,1,36,1,392,2032,404,11,405,-7,406,0.89,407,1,421,1,422,0.5,10,0.5,433,1,490,0.55;")
    .Spawn(0, toggler, False, delay=0.3)
    .Spawn(0, enum.EMPTY2, True)
)

# despawnSetup = (Component("Despawn Setup", unknown_g(), 7)
#     .assert_spawn_order(False)
#     .Spawn(0, 150, False, remap={enum.EMPTY_BULLET: despawner.caller})
#     # .set_context(target=despawner.caller)
#         # .TimerEvent(0, item=enum.EMPTY_TARGET_GROUP, t=0)
#         # .Count(0, item_id=enum.EMPTY_TARGET_GROUP, count=0, activateGroup=True)
#     # .clear_context()
# )

yinyang = EnemyPool(6340, 6379, despawner, num_groups_per=2)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def _enforce_spawn_limit(components: list[ComponentProtocol]) -> None:
    """
    Validate spawn trigger chains to prevent spawn limit bug.

    A spawns B spawns C. We check if B's spawn triggers cause C to be spawn-limited.

    Case 1 (Unmapped): If B has 2+ simultaneous unmapped triggers targeting C,
                       and C has spawn triggers, C gets limited to 1 execution.
                       If C has reset_remap, ALL of B's triggers are treated as unmapped.

    Case 2 (Remapped): If A has a remapped spawn trigger, and B has 2+ simultaneous
                       triggers targeting C, and C has spawn triggers, C gets limited.
                       Exception: If all-but-one of B's simultaneous triggers have
                       reset_remap, they ignore A's remap and don't get limited.
    """
    EXEC_TIME_TOLERANCE = enum.PLR_SPEED / 240  # ~1.298 studs (one tick)

    # Step 1: Build group -> triggers mapping, track spawnOrdered per group
    group_to_triggers: dict[int, list[GenericObj]] = {}
    group_spawn_ordered: dict[int, bool] = {}

    for comp in components:
        group = comp.caller

        if comp.requireSpawnOrder is None:
            warn(f"Component {comp.name} has no spawn order set; defaulting to False")
            comp.assert_spawn_order(False)
        spawn_ordered = bool(comp.requireSpawnOrder)

        if group not in group_spawn_ordered:
            group_spawn_ordered[group] = spawn_ordered
            group_to_triggers[group] = []
        elif group_spawn_ordered[group] != spawn_ordered:
            raise ValueError(
                f"Group {group} has inconsistent spawnOrdered settings across components"
            )

        group_to_triggers[group].extend(comp.triggers)

    # Step 2: Calculate execution time for spawn triggers
    def get_exec_time(trigger: GenericObj, group: int) -> float:
        spawn_ordered = group_spawn_ordered.get(group, False)
        x_pos = float(trigger.get(obj_prop.X, 0)) if spawn_ordered else 0.0
        delay = float(trigger.get(ppt.Spawn.DELAY, 0))
        return x_pos + util.time_to_dist(delay)

    # Step 3: Group spawn triggers by (group, exec_time within tolerance)
    def group_by_exec_time(triggers: list[GenericObj], group: int) -> list[list[GenericObj]]:
        spawn_triggers = [t for t in triggers if t[obj_prop.ID] == obj_id.Trigger.SPAWN]
        if not spawn_triggers:
            return []

        # Sort by exec time
        timed = [(t, get_exec_time(t, group)) for t in spawn_triggers]
        timed.sort(key=lambda x: x[1])

        groups: list[list[GenericObj]] = []
        current_group: list[GenericObj] = [timed[0][0]]
        current_time = timed[0][1]

        for trigger, exec_time in timed[1:]:
            if abs(exec_time - current_time) <= EXEC_TIME_TOLERANCE:
                current_group.append(trigger)
            else:
                groups.append(current_group)
                current_group = [trigger]
                current_time = exec_time

        groups.append(current_group)
        return [g for g in groups if len(g) >= 2]  # Only care about 2+ simultaneous

    # Step 4: Find what groups call this group (A) and what this group calls (C)
    # Use manual caching since these are inner functions
    _find_callers_cache: dict[int, list[tuple[int, GenericObj]]] = {}
    _group_has_spawn_cache: dict[int, bool] = {}
    _c_has_reset_cache: dict[int, bool] = {}

    def find_callers(target_group: int) -> list[tuple[int, GenericObj]]:
        """Find all (group, trigger) pairs where trigger spawns target_group."""
        if target_group in _find_callers_cache:
            return _find_callers_cache[target_group]
        callers: list[tuple[int, GenericObj]] = []
        for group, triggers in group_to_triggers.items():
            for trigger in triggers:
                if trigger[obj_prop.ID] != obj_id.Trigger.SPAWN:
                    continue
                if int(trigger.get(ppt.Move.TARGET_ID, 0)) == target_group:
                    callers.append((group, trigger))
        _find_callers_cache[target_group] = callers
        return callers

    def group_has_spawn_triggers(group: int) -> bool:
        if group in _group_has_spawn_cache:
            return _group_has_spawn_cache[group]
        result = any(
            t[obj_prop.ID] == obj_id.Trigger.SPAWN
            for t in group_to_triggers.get(group, [])
        )
        _group_has_spawn_cache[group] = result
        return result

    def c_has_reset_remap(target_group: int) -> bool:
        """Check if any spawn trigger in target group has reset_remap."""
        if target_group in _c_has_reset_cache:
            return _c_has_reset_cache[target_group]
        result = any(
            t[obj_prop.ID] == obj_id.Trigger.SPAWN and t.get(ppt.Spawn.RESET_REMAP, False)
            for t in group_to_triggers.get(target_group, [])
        )
        _c_has_reset_cache[target_group] = result
        return result

    # Step 5: Run checks for each group B
    for b_group, b_triggers in group_to_triggers.items():
        simultaneous_groups = group_by_exec_time(b_triggers, b_group)

        for sim_triggers in simultaneous_groups:
            # Group by target (C)
            by_target: dict[int, list[GenericObj]] = {}
            for trigger in sim_triggers:
                target = int(trigger.get(ppt.Move.TARGET_ID, 0))
                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(trigger)

            for c_group, triggers_to_c in by_target.items():
                if len(triggers_to_c) < 2: continue
                if not group_has_spawn_triggers(c_group): continue

                # Check if C has reset_remap (treats all B triggers as unmapped)
                c_resets = c_has_reset_remap(c_group)

                unmapped_count = sum(
                    1 for t in triggers_to_c if not t.get(ppt.Spawn.REMAPS, "")
                )

                # Case 1: Check unmapped spawns
                if c_resets:
                    raise RuntimeError(
                        f"Spawn limit violation (Case 1 - C has reset_remap):\n"
                        f"Group {b_group} has {len(triggers_to_c)} simultaneous triggers targeting group {c_group}.\n"
                        f"Group {c_group} has reset_remap, treating all as unmapped.\n"
                        f"Group {c_group} contains spawn trigger(s), causing spawn limit bug."
                    )
                elif unmapped_count >= 2:
                    raise RuntimeError(
                        f"Spawn limit violation (Case 1 - unmapped):\n"
                        f"Group {b_group} has {unmapped_count} simultaneous unmapped triggers targeting group {c_group}.\n"
                        f"Group {c_group} contains spawn trigger(s), causing spawn limit bug."
                    )

                # Case 2: Check if A has remap
                callers = find_callers(b_group)
                a_has_remap = any(
                    caller_trigger.get(ppt.Spawn.REMAPS, "")
                    for _, caller_trigger in callers
                )

                if not a_has_remap: continue

                non_reset_count = sum(
                    1 for t in triggers_to_c
                    if not t.get(ppt.Spawn.RESET_REMAP, False)
                )

                if non_reset_count < 2: continue

                raise RuntimeError(
                    f"Spawn limit violation (Case 2 - A has remap):\n"
                    f"A caller of group {b_group} has a remapped spawn trigger.\n"
                    f"Group {b_group} has {len(triggers_to_c)} simultaneous triggers targeting group {c_group}.\n"
                    f"Only {len(triggers_to_c) - non_reset_count} have reset_remap (need all-but-one).\n"
                    f"Group {c_group} contains spawn trigger(s), causing spawn limit bug."
                )

def _spread_triggers(triggers: list[GenericObj], comp: ComponentProtocol, trigger_area: TriggerArea, len_triggers: int):
    if len_triggers < 1:
        raise ValueError(f"No triggers in component {comp.name}")

    min_x = trigger_area["min_x"]
    max_x = trigger_area["max_x"]
    min_y = trigger_area["min_y"]
    max_y = trigger_area["max_y"]
    if len_triggers == 1:
        triggers[0][obj_prop.X] = random.randint(min_x, max_x)
        triggers[0][obj_prop.Y] = random.randint(min_y, max_y)
        return

    # Single pass to gather all info we need
    first_x = triggers[0].get(obj_prop.X, 0)
    all_same_x = True
    all_keyframe_objs = True
    has_pointer_objs = False
    
    for t in triggers:
        t_x = t.get(obj_prop.X, 0)
        if t[obj_prop.ID] == enum.POINTER_OBJ_ID:
            has_pointer_objs = True
            break
        if t[obj_prop.ID] != obj_id.Trigger.KEYFRAME:
            all_keyframe_objs = False
        if t_x != first_x:
            all_same_x = False
        if not all_keyframe_objs and not all_same_x:
            break

    if has_pointer_objs:
        for pointer_obj in triggers:
            pointer_obj[obj_prop.X] = random.randint(min_x, max_x)
            pointer_obj[obj_prop.Y] = random.randint(min_y, max_y)
        return

    if all_keyframe_objs:
        rand_x = random.randint(min_x, max_x)
        rand_y = random.randint(min_y, max_y)
        for keyframe_obj in triggers:
            keyframe_obj[obj_prop.X] = rand_x
            keyframe_obj[obj_prop.Y] = rand_y
        return

    if all_same_x and not comp.requireSpawnOrder:
        # No spawn order because all_same_x suggests spawn order isnt intended
        for trigger in triggers:
            trigger[obj_prop.X] = random.randint(min_x // 2, max_x // 2) * 2
            trigger[obj_prop.Y] = random.randint(min_y, max_y)
        triggers.sort(key=lambda t: t.get(obj_prop.X, 0))
    elif comp.requireSpawnOrder:
        # Rigid chain - maintain exact spacing (ordered spawn)
        triggers.sort(key=lambda t: t.get(obj_prop.X, 0))
        chain_min_x = triggers[0].get(obj_prop.X, 0)
        chain_max_x = triggers[-1].get(obj_prop.X, 0)
        chain_width = chain_max_x - chain_min_x

        if chain_width > (max_x - min_x):
            raise ValueError(f"Rigid chain too wide ({chain_width}) to fit in trigger area for {comp.name}")

        shift = int(random.randint(min_x, int(max_x - chain_width)) - chain_min_x)
        for trigger in triggers:
            trigger[obj_prop.X] = util.round_to_n_sig_figs(trigger.get(obj_prop.X, 0), 6) + shift
            trigger[obj_prop.Y] = random.randint(min_y, max_y)
    else:
        # Elastic chain - can stretch but must be ordered
        triggers.sort(key=lambda t: t.get(obj_prop.X, 0))
        
        width = (max_x - min_x) / len_triggers
        rand_room = width - 1.3
        
        if width < 1.3:
            raise ValueError(f"Elastic chain too wide to fit in trigger area for {comp.name}")
        
        for i, trigger in enumerate(triggers):
            rand_offset = random.random() * rand_room
            raw_x = min_x + width * i + rand_offset
            trigger[obj_prop.X] = util.round_to_n_sig_figs(raw_x, 6)
            trigger[obj_prop.Y] = random.randint(min_y, max_y)


def _generate_statistics(object_budget: int = 200000) -> dict[str, Any]:
    # Cache trigger counts (single pass over all_components)
    trigger_counts = {comp: len(comp.triggers) for comp in all_components}
    total_triggers = sum(trigger_counts.values())
    
    # Build component usage map and spell stats in one pass over all_spells
    component_usage: dict[ComponentProtocol, int] = {}
    spell_stats = {}
    
    for spell in all_spells:
        spell_trigger_count = 0
        for comp in spell.components:
            component_usage[comp] = component_usage.get(comp, 0) + 1
            spell_trigger_count += trigger_counts[comp]
        spell_stats[spell.spell_name] = spell_trigger_count
    
    # Identify shared components and subtract their counts from spell stats
    shared_trigger_count = 0
    for comp, usage_count in component_usage.items():
        if usage_count > 1:
            comp_triggers = trigger_counts[comp]
            shared_trigger_count += comp_triggers
            # Subtract shared component triggers from each spell that uses it
            for spell in all_spells:
                if comp in spell.components:
                    spell_stats[spell.spell_name] -= comp_triggers
    
    # Build component stats (reuse cached trigger_counts)
    component_stats = {comp.name: (trigger_counts[comp], comp.caller) for comp in all_components}

    return {
        "spell_stats": spell_stats,
        "component_stats": component_stats,
        "shared_trigger_count": shared_trigger_count,
        "budget": {
            "total_triggers": total_triggers,
            "object_budget": object_budget
        }
    }


def _print_budget_analysis(stats: dict[str, Any]) -> None:
    """Print formatted budget analysis to console."""
    budget = stats["budget"]
    remaining_budget = budget["object_budget"] - budget["total_triggers"]
    percentage_used = (budget["total_triggers"] / budget["object_budget"]) * 100
    used_g = unknown_g._level.new.used_group_ids # type: ignore
    used_g_percent = (len(used_g) / 9999) * 100
    print("\n\033[4m=== BUDGET ANALYSIS ===\033[0m")
    print(f"Total triggers: {budget['total_triggers']} out of {remaining_budget} ({percentage_used:.3f}%)")
    print(f"Total groups: {len(used_g)} out of 9999 ({used_g_percent:.3f}%)")

    spell_stats = stats.get("spell_stats", {})
    if spell_stats:
        print("\nSpells:")
        for spell_name, count in spell_stats.items():
            print(f"  {spell_name}: {count} triggers")

    component_stats = stats.get("component_stats", {})
    if component_stats:
        print("\nComponents:")

        max_group_width = max(len(str(group)) for _, (_, group) in component_stats.items())
        max_name_width = max(len(name) for name in component_stats)

        print(f"\033[4m  Group{" " * (max_group_width-2)}"
              f"Name{" "*(max_name_width-3)} Triggers\033[0m")

        for component_name, (count, group) in component_stats.items():
            group_str = f"G{group}".ljust(max_group_width + 1)
            name_str = component_name.ljust(max_name_width)
            print(f"  {group_str}  {name_str}  {count}")

    shared_count = stats.get("shared_trigger_count", 0)
    if shared_count > 0:
        print(f"\nShared components: {shared_count} triggers")


def save_all(*,
    level: Level,
    live_editor: bool = True,
    object_budget: int = 200000,
    check_spawn_limit: bool = True,
    trigger_area: TriggerArea = DEFAULT_TRIGGER_AREA):
    """
    Export all component triggers to JSON file for main.js processing.
    Handles spreading, sorting, validation, and statistics.
    """
    if check_spawn_limit: _enforce_spawn_limit(all_components)

    output: list[GenericObj] = []

    _validate_solid_groups()
    pointer.export_mappings()

    for comp in all_components:
        if comp.current_pc is not None:
            raise RuntimeError(
                f"CRITICAL ERROR: Component {comp.name} has an active pointer circle that has not been cleared yet!"
            )
        
        len_triggers = len(comp.triggers)
        if len_triggers == 0:
            warn(f"Component {comp.name} has no triggers")
            continue

        _spread_triggers(comp.triggers, comp, trigger_area, len_triggers)

        prev_x = -10000
        for trigger in comp.triggers:
            if 9999 in trigger.get(obj_prop.GROUPS,{}):
                raise RuntimeError(
                    f"CRITICAL ERROR: Reserved group 9999 detected in {comp.name}"
                )
            
            if not trigger.get(obj_prop.ID):
                raise RuntimeError(
                    f"CRITICAL ERROR: Trigger in {comp.name} is missing an ID: {trigger}"
                )
            
            curr_x = trigger.get(obj_prop.X, 0)
            if 0 < curr_x - prev_x < 1.28:
                raise RuntimeError(
                    f"CRITICAL ERROR: X position within 1.28 unit of previous trigger"
                    f" in {comp.name} - spawn order not preserved"
                )

            prev_x = curr_x
            output.append(trigger)

    stats = _generate_statistics(object_budget)
    _print_budget_analysis(stats)
    
    level.objects.extend(output)
    if live_editor:
        level.export_to_live_editor()
    else:
        level.export_to_file()