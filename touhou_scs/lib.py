"""
Touhou SCS - Library Module

Core infrastructure: Spell system, GuiderCircles, BulletPools, and export functionality.
Module-level storage for components and spells for automatic registration.
"""

from itertools import cycle
import orjson
import random
import time
import colorsys
from typing import Any, Self

from touhou_scs import enums as enum
from touhou_scs import utils as util
from touhou_scs.component import Component
from touhou_scs.utils import unknown_g, warn
from touhou_scs.types import ComponentProtocol, SpellProtocol, Trigger, TriggerArea
from dataclasses import dataclass

all_spells: list[SpellProtocol] = []
all_components: list[ComponentProtocol] = []

# unfortunately has to be up here to avoid circular imports
solid_groups_to_enforce: set[int] = set()
def _validate_solid_groups(*specific_groups: int):
    """Ensures groups marked as solid are not component or trigger groups."""
    ppt = enum.Properties
    groups = specific_groups if specific_groups else solid_groups_to_enforce

    for g in groups:
        for c in all_components:
            # Check if it's a non-solid group (i.e. if it belongs to a comp/trigger)
            if g in c.groups:
                raise ValueError(f"Group {g} is a component caller in '{c.name}', not a solid group")
            
            for t in c.triggers:
                if t[ppt.OBJ_ID] == enum.ObjectID.POINTER_OBJ: continue
                
                groups = t.get(ppt.GROUPS, [])
                if g in groups:
                    raise ValueError(f"Group {g} is a trigger group ({t[ppt.OBJ_ID]}) in '{c.name}', not a solid group")


_start_time = time.time()

DEFAULT_TRIGGER_AREA: TriggerArea = {
    "min_x": 1350,
    "min_y": 1300,
    "max_x": 9000,
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

class pointer:
    obj_count = 250
    _DEBUG_UI_GROUP = 33
    pointer_comp = Component("Pointers", 0, 11).assert_spawn_order(False)
    pointers = [unknown_g() for _ in range(obj_count)]
    _pointer_iter = cycle(pointers)
    
    group_registry = { g: [g] for g in pointers }
    for g in group_registry: group_registry[g].append(_DEBUG_UI_GROUP)
    
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
            cls.group_registry[p].append(new_group)
    
    @classmethod
    @util.calltracker
    def export_mappings(cls):
        """Call at export time to generate pointer objects with group mappings."""
        if cls.export_mappings.has_been_called:
            raise RuntimeError("Pointer.export_mappings has already been called")
        ppt = enum.Properties
        for group_list in cls.group_registry.values():
            # old pointer obj str: 1,3802,2,2625,3,915,20,2,57,7000.33,64,1,67,1,155,14905,25,9,24,11,128,0.25,129,0.25;
            cls.pointer_comp.triggers.append({ #type: ignore
                ppt.OBJ_ID: enum.ObjectID.POINTER_OBJ,
                ppt.X: 0.0, ppt.Y: 0.0,
                ppt.GROUPS: group_list,
                ppt.EDITOR_LAYER: 11,
                ppt.SCALE: 0.2, #type: ignore
                "24": 9, # Z layer
                "64": True, # dont fade
                "67": True, # dont enter
            })

reimuA_level1 = BulletPool(110, 128, True)

def get_all_components() -> list[ComponentProtocol]: return all_components

class Stage:
    stage1 = Component("Stage1", unknown_g(), 9).assert_spawn_order(True)
    # stage2 = Component("Stage2", unknown_g(), 9).assert_spawn_order(True)
    # stage3 = Component("Stage3", unknown_g(), 9).assert_spawn_order(True)
    # stage4 = Component("Stage4", unknown_g(), 9).assert_spawn_order(True)
    # stage5 = Component("Stage5", unknown_g(), 9).assert_spawn_order(True)
    # stage6 = Component("Stage6", unknown_g(), 9).assert_spawn_order(True)


class EnemyPool:
    def __init__(self, min_group: int, max_group: int, despawn_setup: Component):
        self._min_group = min_group
        self._max_group = max_group
        self._despawn_setup = despawn_setup

        self._current = min_group
        self.__firstcall = True
        self._off_switches = {g: unknown_g() for g in range(min_group, max_group + 1)}

    def next(self) -> int:
        """Cycle to next enemy group in pool"""
        if self.__firstcall:
            self.__firstcall = False
            return self._current

        self._current += 1
        if self._current > self._max_group:
            self._current = self._min_group
        return self._current

    def spawn_enemy(self, stage: Component, time: float, attack: Component, hp: int, enemy_group: int):
        """Spawn an enemy attack with HP/death handling."""
        if not (self._min_group <= enemy_group <= self._max_group):
            raise ValueError(
                f"spawn_enemy: enemy_group {enemy_group} is not in pool range "
                f"{self._min_group}-{self._max_group}"
            )

        util.enforce_component_targets("Spawn Enemy", attack,
            excludes={ enum.EMPTY_BULLET, enum.EMPTY1, enum.EMPTY2, enum.EMPTY_EMITTER, enum.EMPTY_MULTITARGET, enum.EMPTY_TARGET_GROUP })

        off_switch = self._off_switches[enemy_group]

        with stage.temp_context(groups=off_switch):
            stage.Spawn(time, attack.caller, True)

        stage.Spawn(time, self._despawn_setup.caller, False,
            remap=f"{enum.EMPTY_TARGET_GROUP}.{enemy_group}.{enum.EMPTY1}.{off_switch}")
        stage.Pickup(time - enum.TICK*2, item_id=enemy_group, count=hp, override=True)


# less annoying way instead of making 'despawner' have spawn order (uses spawn delay instead)
toggler = (Component("Toggler", unknown_g(), 7)
    .assert_spawn_order(False)
    .set_context(target=enum.EMPTY_TARGET_GROUP)
        .Toggle(0, False)
    .clear_context()
)

despawner = (Component("Despawner", unknown_g(), 7)
    .assert_spawn_order(False)
    .set_context(target=enum.EMPTY_TARGET_GROUP)
        .Alpha(0, t=1, opacity=0)
        .Pulse(0, HSB(0, 0, -20), fadeIn=0.1, t=0.3, fadeOut=0.6, exclusive=True)
        .Scale(0, factor=0.1, t=0.5, hold=3)
    .clear_context()
    .Stop(0, target=enum.EMPTY1)
    .Spawn(0, toggler, False, delay=1)
)

despawnSetup = (Component("Despawn Setup", unknown_g(), 7)
    .assert_spawn_order(False)
    .set_context(target=despawner.caller)
        .Count(0, item_id=enum.EMPTY_TARGET_GROUP, count=0, activateGroup=True)
    .clear_context()
)

enemy1 = EnemyPool(200, 211, despawnSetup)

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
    ppt = enum.Properties
    EXEC_TIME_TOLERANCE = enum.PLR_SPEED / 240  # ~1.298 studs (one tick)

    # Step 1: Build group -> triggers mapping, track spawnOrdered per group
    group_to_triggers: dict[int, list[Trigger]] = {}
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
    def get_exec_time(trigger: Trigger, group: int) -> float:
        spawn_ordered = group_spawn_ordered.get(group, False)
        x_pos = float(trigger.get(ppt.X, 0)) if spawn_ordered else 0.0
        delay = float(trigger.get(ppt.SPAWN_DELAY, 0))
        return x_pos + util.time_to_dist(delay)

    # Step 3: Group spawn triggers by (group, exec_time within tolerance)
    def group_by_exec_time(triggers: list[Trigger], group: int) -> list[list[Trigger]]:
        spawn_triggers = [t for t in triggers if t[ppt.OBJ_ID] == enum.ObjectID.SPAWN]
        if not spawn_triggers:
            return []

        # Sort by exec time
        timed = [(t, get_exec_time(t, group)) for t in spawn_triggers]
        timed.sort(key=lambda x: x[1])

        groups: list[list[Trigger]] = []
        current_group: list[Trigger] = [timed[0][0]]
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
    _find_callers_cache: dict[int, list[tuple[int, Trigger]]] = {}
    _group_has_spawn_cache: dict[int, bool] = {}
    _c_has_reset_cache: dict[int, bool] = {}

    def find_callers(target_group: int) -> list[tuple[int, Trigger]]:
        """Find all (group, trigger) pairs where trigger spawns target_group."""
        if target_group in _find_callers_cache:
            return _find_callers_cache[target_group]
        callers: list[tuple[int, Trigger]] = []
        for group, triggers in group_to_triggers.items():
            for trigger in triggers:
                if trigger[ppt.OBJ_ID] != enum.ObjectID.SPAWN:
                    continue
                if int(trigger.get(ppt.TARGET, 0)) == target_group:
                    callers.append((group, trigger))
        _find_callers_cache[target_group] = callers
        return callers

    def group_has_spawn_triggers(group: int) -> bool:
        if group in _group_has_spawn_cache:
            return _group_has_spawn_cache[group]
        result = any(
            t[ppt.OBJ_ID] == enum.ObjectID.SPAWN
            for t in group_to_triggers.get(group, [])
        )
        _group_has_spawn_cache[group] = result
        return result

    def c_has_reset_remap(target_group: int) -> bool:
        """Check if any spawn trigger in target group has reset_remap."""
        if target_group in _c_has_reset_cache:
            return _c_has_reset_cache[target_group]
        result = any(
            t[ppt.OBJ_ID] == enum.ObjectID.SPAWN and t.get(ppt.RESET_REMAP, False)
            for t in group_to_triggers.get(target_group, [])
        )
        _c_has_reset_cache[target_group] = result
        return result

    # Step 5: Run checks for each group B
    for b_group, b_triggers in group_to_triggers.items():
        simultaneous_groups = group_by_exec_time(b_triggers, b_group)

        for sim_triggers in simultaneous_groups:
            # Group by target (C)
            by_target: dict[int, list[Trigger]] = {}
            for trigger in sim_triggers:
                target = int(trigger.get(ppt.TARGET, 0))
                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(trigger)

            for c_group, triggers_to_c in by_target.items():
                if len(triggers_to_c) < 2: continue
                if not group_has_spawn_triggers(c_group): continue

                # Check if C has reset_remap (treats all B triggers as unmapped)
                c_resets = c_has_reset_remap(c_group)

                unmapped_count = sum(
                    1 for t in triggers_to_c if not t.get(ppt.REMAP_STRING, "")
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
                    caller_trigger.get(ppt.REMAP_STRING, "")
                    for _, caller_trigger in callers
                )

                if not a_has_remap: continue

                non_reset_count = sum(
                    1 for t in triggers_to_c
                    if not t.get(ppt.RESET_REMAP, False)
                )

                if non_reset_count < 2: continue

                raise RuntimeError(
                    f"Spawn limit violation (Case 2 - A has remap):\n"
                    f"A caller of group {b_group} has a remapped spawn trigger.\n"
                    f"Group {b_group} has {len(triggers_to_c)} simultaneous triggers targeting group {c_group}.\n"
                    f"Only {len(triggers_to_c) - non_reset_count} have reset_remap (need all-but-one).\n"
                    f"Group {c_group} contains spawn trigger(s), causing spawn limit bug."
                )

def _spread_triggers(triggers: list[Trigger], comp: ComponentProtocol, trigger_area: TriggerArea, len_triggers: int):
    if len_triggers < 1:
        raise ValueError(f"No triggers in component {comp.name}")

    min_x = trigger_area["min_x"]
    max_x = trigger_area["max_x"]
    min_y = trigger_area["min_y"]
    max_y = trigger_area["max_y"]
    ppt = enum.Properties

    if len_triggers == 1:
        triggers[0][ppt.X] = random.randint(min_x, max_x)
        triggers[0][ppt.Y] = random.randint(min_y, max_y)
        return

    # Single pass to gather all info we need
    first_x = triggers[0][ppt.X]
    all_same_x = True
    all_keyframe_objs = True
    has_pointer_objs = False
    
    for t in triggers:
        t_x = t[ppt.X]
        if t[ppt.OBJ_ID] == enum.ObjectID.POINTER_OBJ:
            has_pointer_objs = True
            break
        if t[ppt.OBJ_ID] != enum.ObjectID.KEYFRAME_OBJ:
            all_keyframe_objs = False
        if t_x != first_x:
            all_same_x = False
        if not all_keyframe_objs and not all_same_x:
            break

    if has_pointer_objs:
        for pointer_obj in triggers:
            pointer_obj[ppt.X] = random.randint(min_x, max_x)
            pointer_obj[ppt.Y] = random.randint(min_y, max_y)
        return

    if all_keyframe_objs:
        rand_x = random.randint(min_x, max_x)
        rand_y = random.randint(min_y, max_y)
        for keyframe_obj in triggers:
            keyframe_obj[ppt.X] = rand_x
            keyframe_obj[ppt.Y] = rand_y
        return

    if all_same_x and not comp.requireSpawnOrder:
        # No spawn order because all_same_x suggests spawn order isnt intended
        for trigger in triggers:
            trigger[ppt.X] = random.randint(min_x // 2, max_x // 2) * 2
            trigger[ppt.Y] = random.randint(min_y, max_y)
        triggers.sort(key=lambda t: t[ppt.X])
    elif comp.requireSpawnOrder:
        # Rigid chain - maintain exact spacing (ordered spawn)
        triggers.sort(key=lambda t: t[ppt.X])
        chain_min_x = triggers[0][ppt.X]
        chain_max_x = triggers[-1][ppt.X]
        chain_width = chain_max_x - chain_min_x

        if chain_width > (max_x - min_x):
            raise ValueError(f"Rigid chain too wide ({chain_width}) to fit in trigger area for {comp.name}")

        shift = int(random.randint(min_x, int(max_x - chain_width)) - chain_min_x)
        for trigger in triggers:
            trigger[ppt.X] = util.round_to_n_sig_figs(trigger[ppt.X], 6) + shift
            trigger[ppt.Y] = random.randint(min_y, max_y)
    else:
        # Elastic chain - can stretch but must be ordered
        triggers.sort(key=lambda t: t[ppt.X])
        
        width = (max_x - min_x) / len_triggers
        rand_room = width - 1.3
        
        if width < 1.3:
            raise ValueError(f"Elastic chain too wide to fit in trigger area for {comp.name}")
        
        for i, trigger in enumerate(triggers):
            rand_offset = random.random() * rand_room
            raw_x = min_x + width * i + rand_offset
            trigger[ppt.X] = util.round_to_n_sig_figs(raw_x, 6)
            trigger[ppt.Y] = random.randint(min_y, max_y)


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
    print("\n\033[4m=== BUDGET ANALYSIS ===\033[0m")
    print(f"Total triggers: {budget['total_triggers']} ({percentage_used:.3f}%)")
    print(f"Remaining budget: {remaining_budget} triggers")

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
    filename: str = "triggers.json",
    object_budget: int = 200000,
    check_spawn_limit: bool = True,
    trigger_area: TriggerArea = DEFAULT_TRIGGER_AREA):
    """
    Export all component triggers to JSON file for main.js processing.
    Handles spreading, sorting, validation, and statistics.
    """
    if check_spawn_limit: _enforce_spawn_limit(all_components)

    output: dict[str, list[Trigger]] = {"triggers": []}

    ppt = enum.Properties # shorthand (also technically faster lol)
    
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
            if 9999 in trigger[ppt.GROUPS]:
                raise RuntimeError(
                    f"CRITICAL ERROR: Reserved group 9999 detected in {comp.name}"
                )
            
            curr_x = trigger[ppt.X]
            if 0 < curr_x - prev_x < 1.28:
                raise RuntimeError(
                    f"CRITICAL ERROR: X position within 1.28 unit of previous trigger"
                    f" in {comp.name} - spawn order not preserved"
                )

            prev_x = curr_x
            output["triggers"].append(trigger)

    stats = _generate_statistics(object_budget)
    _print_budget_analysis(stats)

    if filename == "testing": return

    with open(filename, "wb") as file:
        file.write(orjson.dumps(output))

    elapsed = time.time() - _start_time
    print(f"\nSaved to {filename} successfully!")
    print(f"Total execution time: {elapsed:.3f} seconds")
