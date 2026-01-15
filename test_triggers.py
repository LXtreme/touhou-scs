
# pyright: reportTypedDictNotRequiredAccess=false
# pyright: reportArgumentType=false
# pyright: reportPrivateUsage=false
"""
Focused unit tests for trigger methods in Component class.

Philosophy: Tests should expose logic bugs and verify validation.
- Test parameter boundaries that should be rejected
- Test edge cases at validation limits
- Verify error messages are correct
- Skip trivial tests that just verify property assignment
"""

import warnings
import pytest
from pytest import ExceptionInfo
from touhou_scs.component import Component, Multitarget
from touhou_scs import enums, lib, utils
from typing import Any
from touhou_scs.movements import CurveType


@pytest.fixture(autouse=True)
def reset_global_state():
    """Clear global state before each test to prevent interference."""
    lib.all_components.clear()
    lib.solid_groups_to_enforce.clear()
    yield


def setup_pointer_circle(caller: Component) -> Component:
    """Helper to set up a PointerCircle context for pattern tests."""
    caller.assert_spawn_order(True)
    caller.pointer.SetPointerCircle(0, location=100, follow=False)
    return caller


@pytest.fixture
def comp_with_target():
    """Component with target pre-set for simple tests."""
    comp = Component("Test", 100)
    comp.set_context(target=50)
    return comp


@pytest.fixture
def spawn_ordered_comp():
    """Component with spawn order and standard empty targets set up."""
    comp = Component("Test", 100).assert_spawn_order(True)
    comp.set_context(target=enums.EMPTY_BULLET)
    comp.Toggle(0, activateGroup=True)
    comp.set_context(target=enums.EMPTY_TARGET_GROUP)
    comp.Toggle(0, activateGroup=True)
    return comp


ppt = enums.Properties

def assert_error(exc_info: ExceptionInfo[BaseException], *patterns: str) -> None:
    """Assert exception message contains all patterns (case-insensitive)."""
    msg = str(exc_info.value).lower()
    for pattern in patterns:
        p = pattern.lower()
        assert p in msg, f"Expected '{pattern}' in: {str(exc_info.value)}"

def assert_warning(warning_list: list[warnings.WarningMessage], *patterns: str) -> None:
    """Assert warning list contains all patterns (case-insensitive)."""
    combined_msgs = " ".join(str(w.message).lower() for w in warning_list)
    for pattern in patterns:
        p = pattern.lower()
        assert p in combined_msgs, f"Expected '{pattern}' in warnings."

# ============================================================================
# SPAWN TRIGGER - Target Group Validation
# ============================================================================

class TestSpawnTargetValidation:
    def _test_spawn_target(self, target: int, should_fail: bool, *error_patterns: str):
        """Helper to test Spawn with different target values"""
        comp = Component("Test", 100)
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp.Spawn(0, target, spawnOrdered=False)
            assert_error(exc, *error_patterns)
        else:
            comp.Spawn(0, target, spawnOrdered=False)
            assert comp.triggers[0][ppt.TARGET] == target
    
    @pytest.mark.parametrize("target,error_patterns", [
        (0, ("positive", "0")),
        (-50, ("positive", "-50")),
        (3, ("restricted", "3")),
        (utils.unknown_g.counter + 1, ("out of valid range",)),
    ])
    def test_spawn_target_rejected(self, target: int, error_patterns: tuple[str, ...]):
        self._test_spawn_target(target, True, *error_patterns)
    
    @pytest.mark.parametrize("target", [10, utils.unknown_g.counter])
    def test_spawn_target_valid(self, target: int):
        self._test_spawn_target(target, False)


class TestSpawnDelayValidation:
    @pytest.mark.parametrize("delay,should_fail,error_patterns", [
        (-1, True, ("non-negative", "-1")),
        (0, False, ()),
        (0.5, False, ()),
    ])
    def test_spawn_delay(self, delay: float, should_fail: bool, error_patterns: tuple[str, ...]):
        comp = Component("Test", 100)
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp.Spawn(0, 50, spawnOrdered=False, delay=delay)
            assert_error(exc, *error_patterns)
        else:
            comp.Spawn(0, 50, spawnOrdered=False, delay=delay)
            if delay > 0:
                assert comp.triggers[0][ppt.SPAWN_DELAY] == delay
    
    def test_spawn_zero_delay_not_stored(self):
        """Zero delay is not stored as a property"""
        comp = Component("Test", 100).Spawn(0, 50, spawnOrdered=False, delay=0)
        assert ppt.SPAWN_DELAY not in comp.triggers[0]
    
    def test_spawn_positive_delay_stored(self):
        """Positive delay is stored correctly"""
        comp = Component("Test", 100).Spawn(0, 50, spawnOrdered=False, delay=0.5)
        assert comp.triggers[0][ppt.SPAWN_DELAY] == 0.5


class TestSpawnRemapValidation:
    def test_spawn_remap_empty_string_not_stored(self):
        """Empty remap string is silently skipped"""
        comp = Component("Test", 100).Spawn(0, 50, spawnOrdered=False, remap="")
        assert ppt.REMAP_STRING not in comp.triggers[0]

    def test_spawn_remap_odd_pairs_rejected(self):
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 50, spawnOrdered=False, remap="1.2.3")
        assert_error(exc, "even number")

    def test_spawn_remap_duplicate_source_rejected(self):
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 50, spawnOrdered=False, remap="10.20.10.30")
        assert_error(exc, "duplicate source", "10")


# ============================================================================
# MOVE TRIGGERS - Easing Boundaries
# ============================================================================

class TestMoveEasingValidation:
    @pytest.mark.parametrize("type,rate,should_fail,error_patterns", [
        (-1, None, True, ("type", "-1")),
        (19, None, True, ("type", "19")),
        (2.5, None, True, ("type", "2.5")),  # Non-integer float
        (0, None, False, ()),
        (18, None, False, ()),
        (None, 0.10, True, ("rate", "0.1")),
        (None, 0.05, True, ("rate", "0.05")),
        (None, 20.01, True, ("rate", "20.01")),
        (None, 0.11, False, ()),
        (None, 20.0, False, ()),
    ])
    def test_easing_validation(self, comp_with_target: Component, type: int | None, rate: float | None, should_fail: bool, error_patterns: tuple[str, ...]):
        kwargs = {"t": 1.0, "dist": 100}
        if type is not None:
            kwargs["type"] = type
        if rate is not None:
            kwargs["rate"] = rate
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp_with_target.MoveTowards(0, targetDir=60, **kwargs)
            assert_error(exc, *error_patterns)
        else:
            comp_with_target.MoveTowards(0, targetDir=60, **kwargs)
            if type is not None:
                assert comp_with_target.triggers[0][ppt.EASING] == type
            if rate is not None:
                assert comp_with_target.triggers[0][ppt.EASING_RATE] == rate


class TestMoveDurationValidation:
    def test_duration_negative_rejected(self, comp_with_target: Component):
        with pytest.raises(ValueError) as exc:
            comp_with_target.MoveTowards(0, targetDir=60, t=-0.5, dist=100)
        assert_error(exc, "non-negative", "-0.5")
    
    def test_duration_zero_sets_silent(self, comp_with_target: Component):
        """Duration=0 should set MOVE_SILENT=True"""
        comp_with_target.MoveTowards(0, targetDir=60, t=0, dist=100)
        assert comp_with_target.triggers[0][ppt.MOVE_SILENT]
    
    def test_duration_positive_no_silent(self, comp_with_target: Component):
        """Duration>0 should not set MOVE_SILENT"""
        comp_with_target.MoveTowards(0, targetDir=60, t=1.0, dist=100)
        assert ppt.MOVE_SILENT not in comp_with_target.triggers[0]


# ============================================================================
# ALPHA TRIGGER - Opacity Boundaries
# ============================================================================

class TestAlphaOpacityValidation:
    @pytest.mark.parametrize("opacity,should_fail,error_patterns", [
        (-1, True, ("between 0 and 100",)),
        (101, True, ("between 0 and 100",)),
        (0, False, ()),
        (50, False, ()),
        (100, False, ()),
    ])
    def test_opacity_validation(self, comp_with_target: Component, opacity: float, should_fail: bool, error_patterns: tuple[str, ...]):
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp_with_target.Alpha(0, opacity=opacity)
            assert_error(exc, *error_patterns)
        else:
            comp_with_target.Alpha(0, opacity=opacity)
            # Opacity is stored as decimal (0-1 range)
            assert comp_with_target.triggers[0][ppt.OPACITY] == opacity / 100.0
    
    def test_opacity_converts_to_decimal(self, comp_with_target: Component):
        """Opacity is stored as decimal (0-1 range)"""
        comp_with_target.Alpha(0, opacity=75)
        assert comp_with_target.triggers[0][ppt.OPACITY] == 0.75


# ============================================================================
# SCALE TRIGGER - Factor Validation
# ============================================================================

class TestScaleFactorValidation:
    @pytest.mark.parametrize("factor,hold,should_fail,error_patterns", [
        (0, 0.5, True, ("factor", ">0", "0")),
        (-1, 0.5, True, ("factor", ">0", "-1")),
        (1.0, 0.5, True, ("1", "has no effect")),
        (2.0, -0.1, True, ("non-negative", "-0.1")),
        (0.9999, 0.5, False, ()),
        (1.0001, 0.5, False, ()),
        (2.0, 0.5, False, ()),
    ])
    def test_scale_validation(self, comp_with_target: Component, factor: float, hold: float, should_fail: bool, error_patterns: tuple[str, ...]):
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp_with_target.Scale(0, factor=factor, t=1.0, hold=hold)
            assert_error(exc, *error_patterns)
        else:
            # May produce warnings for factors near 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comp_with_target.Scale(0, factor=factor, t=1.0, hold=hold)
            assert len(comp_with_target.triggers) > 0


# ============================================================================
# COUNT TRIGGER - Item ID Validation
# ============================================================================


class TestCountItemIdValidation:
    @pytest.mark.parametrize("item_id,should_fail,error_patterns", [
        (0, True, ("positive", "0")),
        (-1, True, ("positive", "-1")),
        (10000, True, ("positive", "10000")),
        (1, False, ()),
        (9999, False, ()),
    ])
    def test_count_item_id_validation(self, comp_with_target: Component, item_id: int, should_fail: bool, error_patterns: tuple[str, ...]):
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp_with_target.Count(0, item_id=item_id, count=5, activateGroup=True)
            assert_error(exc, *error_patterns)
        else:
            comp_with_target.Count(0, item_id=item_id, count=5, activateGroup=True)
            assert comp_with_target.triggers[0][ppt.ITEM_ID] == item_id


# ============================================================================
# PICKUP TRIGGER - Validation
# ============================================================================

class TestPickupValidation:
    @pytest.mark.parametrize("item_id,count,should_fail,error_patterns", [
        (0, None, True, ("positive", "0")),
        (-1, None, True, ("positive", "-1")),
        (10000, None, True, ("positive", "10000")),
        (None, 0, True, ("no change", "0")),
        (1, None, False, ()),
        (9999, None, False, ()),
        (None, -10, False, ()),
    ])
    def test_pickup_validation(self, item_id: int | None, count: int | None, should_fail: bool, error_patterns: tuple[str, ...]):
        comp = Component("Test", 100)
        # Use defaults for unspecified params
        if item_id is None:
            item_id = 5
        if count is None:
            count = 50
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp.Pickup(0, item_id=item_id, count=count, override=False)
            assert_error(exc, *error_patterns)
        else:
            comp.Pickup(0, item_id=item_id, count=count, override=False)
            assert comp.triggers[0][ppt.ITEM_ID] == item_id
            assert comp.triggers[0][ppt.PICKUP_COUNT] == count
    
    def test_pickup_no_target_property(self):
        """Pickup trigger should not have TARGET property"""
        comp = Component("Test", 100)
        comp.Pickup(0, item_id=12, count=50, override=True)
        assert ppt.TARGET not in comp.triggers[0]


# ============================================================================
# PICKUP MODIFY TRIGGER - Validation
# ============================================================================

class TestPickupModifyValidation:
    def _test_pickup_modify(self, item_id: int = 5, factor: float = 1.5,
                            multiply: bool = False, divide: bool = False,
                            should_fail: bool = False, error_patterns: tuple[str, ...] = ()):
        """Helper to test PickupModify with different parameters"""
        comp = Component("Test", 100)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                comp.PickupModify(0, item_id=item_id, factor=factor, multiply=multiply, divide=divide)  # type: ignore
            assert_error(exc, *error_patterns)
        else:
            comp.PickupModify(0, item_id=item_id, factor=factor, multiply=multiply, divide=divide)  # type: ignore
            assert len(comp.triggers) > 0
    
    @pytest.mark.parametrize("item_id,error_patterns", [
        (0, ("positive", "0")),
        (10000, ("positive", "10000")),
    ])
    def test_pickup_modify_item_id_rejected(self, item_id: int, error_patterns: tuple[str, ...]):
        self._test_pickup_modify(item_id=item_id, multiply=True, should_fail=True, error_patterns=error_patterns)

    @pytest.mark.parametrize("factor", [1, 1.0])
    def test_pickup_modify_factor_one_rejected(self, factor: float):
        self._test_pickup_modify(factor=factor, multiply=True, should_fail=True, error_patterns=("1 has no effect",))

    def test_pickup_modify_no_mode_rejected(self):
        self._test_pickup_modify(should_fail=True, error_patterns=("multiply", "divide"))

    def test_pickup_modify_both_modes_rejected(self):
        self._test_pickup_modify(multiply=True, divide=True, should_fail=True, error_patterns=("both", "multiply", "divide"))
    
    def test_pickup_modify_valid_modes_accepted(self):
        self._test_pickup_modify(factor=2.0, multiply=True, should_fail=False)
        self._test_pickup_modify(factor=2.0, divide=True, should_fail=False)
    
    def test_pickup_modify_multiply_mode_value(self):
        """Multiply mode should set PICKUP_MULTIPLY_DIVIDE=1"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=11, factor=1.45, multiply=True)
        assert comp.triggers[0][ppt.PICKUP_MULTIPLY_DIVIDE] == 1
    
    def test_pickup_modify_divide_mode_value(self):
        """Divide mode should set PICKUP_MULTIPLY_DIVIDE=2"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=11, factor=2.0, divide=True)
        assert comp.triggers[0][ppt.PICKUP_MULTIPLY_DIVIDE] == 2
    
    def test_pickup_modify_no_target_property(self):
        """PickupModify trigger should not have TARGET property"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=11, factor=1.5, multiply=True)
        assert ppt.TARGET not in comp.triggers[0]


# ============================================================================
# ROTATE / POINT TO GROUP - Dynamic + Easing
# ============================================================================

class TestPointToGroupValidation:
    def test_point_to_group_dynamic_with_easing_type_rejected(self, comp_with_target: Component):
        with pytest.raises(ValueError) as exc:
            comp_with_target.PointToGroup(0, targetDir=60, dynamic=True, type=1)
        assert_error(exc, "dynamic", "easing")

    def test_point_to_group_dynamic_with_easing_rate_rejected(self, comp_with_target: Component):
        with pytest.raises(ValueError) as exc:
            comp_with_target.PointToGroup(0, targetDir=60, dynamic=True, rate=1.5)
        assert_error(exc, "dynamic", "easing")
    
    def test_point_to_group_dynamic_without_easing_valid(self, comp_with_target: Component):
        """Dynamic mode without easing params should work"""
        comp_with_target.PointToGroup(0, targetDir=60, dynamic=True)
        assert len(comp_with_target.triggers) == 1


class TestRotateValidation:
    def test_rotate_center_defaults_to_target(self, comp_with_target: Component):
        """Center defaults to target when not specified"""
        comp_with_target.Rotate(0, angle=45, t=1.0)
        trigger = comp_with_target.triggers[0]
        assert trigger[ppt.ROTATE_CENTER] == 50

    def test_rotate_restricted_center_rejected(self, comp_with_target: Component):
        with pytest.raises(ValueError) as exc:
            comp_with_target.Rotate(0, angle=45, center=3, t=1.0)
        assert_error(exc, "restricted", "3")


# ============================================================================
# GROUP CONTEXT - State Management
# ============================================================================

@pytest.fixture
def context_comp():
    """Fixture providing a fresh Component for context tests"""
    return Component("Test", 100)

class TestGroupContextManagement:
    def test_start_context_adds_groups_to_subsequent_triggers(self, context_comp: Component):
        """Triggers after set_context include context groups"""
        context_comp.set_context(groups=200, target=50)
        context_comp.Toggle(0, activateGroup=True)
        assert 200 in context_comp.triggers[0][ppt.GROUPS]

    def test_end_context_removes_context_groups(self, context_comp: Component):
        """Triggers after clear_context exclude context groups"""
        context_comp.set_context(groups=200, target=50)
        context_comp.Toggle(0, activateGroup=True)
        context_comp.clear_context(groups_only=True)
        context_comp.set_context(target=51)
        context_comp.Toggle(0.1, activateGroup=True)

        assert 200 in context_comp.triggers[0][ppt.GROUPS]
        assert 200 not in context_comp.triggers[1][ppt.GROUPS]

    def test_nested_context_overwrites(self, context_comp: Component):
        """Setting groups context while one is active overwrites it"""
        context_comp.set_context(groups=200, target=50)
        context_comp.Toggle(0, activateGroup=True)
        context_comp.set_context(groups=300, target=51)
        context_comp.Toggle(0.1, activateGroup=True)

        assert 200 in context_comp.triggers[0][ppt.GROUPS]
        assert 200 not in context_comp.triggers[1][ppt.GROUPS]
        assert 300 in context_comp.triggers[1][ppt.GROUPS]

    def test_clear_context_is_idempotent(self, context_comp: Component):
        """Clearing context without setting is safe (idempotent)"""
        context_comp.clear_context(groups_only=True)
        context_comp.clear_context()
        assert context_comp.groups == [context_comp.caller]
        assert context_comp.target == -1

    def test_context_with_multiple_groups(self, context_comp: Component):
        """Context can add multiple groups at once"""
        context_comp.set_context(groups=[200, 201, 202], target=50)
        context_comp.Toggle(0, activateGroup=True)
        assert all(g in context_comp.triggers[0][ppt.GROUPS] for g in [200, 201, 202])

    def test_context_with_list_groups(self, context_comp: Component):
        """Context accepts list of groups"""
        context_comp.set_context(groups=[200, 201], target=50)
        context_comp.Toggle(0, activateGroup=True)
        assert 200 in context_comp.triggers[0][ppt.GROUPS]
        assert 201 in context_comp.triggers[0][ppt.GROUPS]

    def test_context_empty_rejected(self, context_comp: Component):
        """Setting context with no parameters is rejected"""
        with pytest.raises(ValueError) as exc:
            context_comp.set_context()
        assert_error(exc, "must provide target or groups")


class TestTempContext:
    """Test temp_context contextmanager"""

    def test_temp_context_restores_target(self):
        """temp_context restores target after exit"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)

        with comp.temp_context(target=60):
            comp.Toggle(0.1, activateGroup=True)

        assert comp.target == 50
        assert comp.triggers[0][ppt.TARGET] == 50
        assert comp.triggers[1][ppt.TARGET] == 60

    def test_temp_context_restores_both(self):
        """temp_context can temporarily change both target and groups"""
        comp = Component("Test", 100)
        comp.set_context(target=50, groups=200)

        with comp.temp_context(target=60, groups=300):
            comp.Toggle(0, activateGroup=True)

        assert comp.target == 50
        assert comp.groups == [100, 200]

    def test_temp_context_nesting(self):
        """temp_context supports nesting"""
        comp = Component("Test", 100)
        comp.set_context(target=50, groups=200)

        with comp.temp_context(groups=300):
            comp.Toggle(0, activateGroup=True)
            with comp.temp_context(target=70):
                comp.Toggle(0.1, activateGroup=True)
            # After inner context: target=50, groups=300
            assert comp.target == 50
            assert comp.groups == [100, 300]

        # After outer context: fully restored
        assert comp.target == 50
        assert comp.groups == [100, 200]

    def test_temp_context_with_no_params(self):
        """temp_context with no params is a no-op but still works"""
        comp = Component("Test", 100)
        comp.set_context(target=50, groups=200)

        with pytest.raises(ValueError) as exc:
            with comp.temp_context():
                comp.Toggle(0, activateGroup=True)
        assert_error(exc, "must provide target or groups")


class TestFlattenGroups:
    """Test _flatten_groups duplicate detection"""

    def test_flatten_detects_duplicates_in_single_list(self):
        comp = Component("Test", 100)
        with pytest.raises(ValueError, match="Duplicate"):
            comp._flatten_groups([100, 200, 100])

    def test_flatten_detects_duplicates_across_args(self):
        comp = Component("Test", 100)
        with pytest.raises(ValueError, match="Duplicate"):
            comp._flatten_groups(100, [200, 100])


# ============================================================================
# INSTANT PATTERNS - Spawn Order Requirement
# ============================================================================

class TestInstantPatternSpawnOrderRequirement:
    def test_arc_without_spawn_order_rejected(self, spawn_ordered_comp: Component):
        """Arc requires an active PointerCircle"""
        caller = Component("Caller", 200).assert_spawn_order(True)
        # No SetPointerCircle called - should fail
        with pytest.raises(RuntimeError) as exc:
            caller.instant.Arc(
                time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                numBullets=5, angle=150
            )
        assert_error(exc, "No active PointerCircle")

    def test_radial_without_spawn_order_rejected(self, spawn_ordered_comp: Component):
        """Radial requires an active PointerCircle"""
        caller = Component("Caller", 200).assert_spawn_order(True)
        # No SetPointerCircle called - should fail
        with pytest.raises(RuntimeError) as exc:
            caller.instant.Radial(
                time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                spacing=30
            )
        assert_error(exc, "No active PointerCircle")

    def test_line_without_spawn_order_rejected(self):
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.instant.Line(
                time=0, comp=comp, emitter=50 ,targetDir=90, bullet=lib.bullet2,
                numBullets=5, fastestTime=0.5, slowestTime=2.0, dist=100
            )
        assert_error(exc, "must require spawn order")


# ============================================================================
# INSTANT ARC PATTERN - Complex Validation Logic
# ============================================================================

class TestInstantArcValidation:
    @pytest.mark.parametrize("angle,numBullets,should_fail,error_patterns", [
        (0, 5, True, ("angle", "0")),
        (361, 5, True, ("angle", "360")),
        (150, 5, False, ()),
        (360, 5, False, ()),
    ])
    def test_arc_validation(self, spawn_ordered_comp: Component, angle: float, numBullets: int, should_fail: bool, error_patterns: tuple[str, ...]):
        # Setup caller with pointer circle
        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                caller.instant.Arc(
                    time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                    numBullets=numBullets, angle=angle
                )
            assert_error(exc, *error_patterns)
        else:
            caller.instant.Arc(
                time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                numBullets=numBullets, angle=angle
            )
            assert len(caller.triggers) > 0


# ============================================================================
# INSTANT RADIAL PATTERN - Validation
# ============================================================================


class TestInstantRadialValidation:
    def _test_instant_radial(self, spawn_ordered_comp: Component, spacing: float | None = None, numBullets: int | None = None,
                             should_fail: bool = False, error_patterns: tuple[str, ...] = ()):
        """Helper to test instant.Radial with different parameters"""
        # Setup caller with pointer circle
        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                caller.instant.Radial(  # type: ignore
                    time=0, comp=spawn_ordered_comp, bullet=lib.bullet1, spacing=spacing, numBullets=numBullets
                )
            assert_error(exc, *error_patterns)
        else:
            caller.instant.Radial(  # type: ignore
                time=0, comp=spawn_ordered_comp, bullet=lib.bullet1, spacing=spacing, numBullets=numBullets
            )
            assert len(caller.triggers) > 0
    
    def test_radial_neither_spacing_nor_numbullets_rejected(self, spawn_ordered_comp: Component):
        self._test_instant_radial(spawn_ordered_comp, should_fail=True, error_patterns=("must provide", "spacing", "numBullets"))

    def test_radial_mismatched_spacing_and_numbullets_rejected(self, spawn_ordered_comp: Component):
        self._test_instant_radial(spawn_ordered_comp, numBullets=12, spacing=20, should_fail=True, error_patterns=("don't match",))
    
    @pytest.mark.parametrize("spacing,numBullets", [
        (30, None),
        (None, 12),
    ])
    def test_radial_valid_patterns_accepted(self, spawn_ordered_comp: Component, spacing: float | None, numBullets: int | None):
        self._test_instant_radial(spawn_ordered_comp, spacing=spacing, numBullets=numBullets, should_fail=False)


# ============================================================================
# INSTANT LINE PATTERN - Validation
# ============================================================================

class TestInstantLineValidation:
    @pytest.mark.parametrize("numBullets,fastestTime,slowestTime,should_fail,error_patterns", [
        (5, 0, 2.0, True, ("positive", "0")),
        (5, 2.0, 1.0, True, ("greater than", "2.0", "1.0")),
        (2, 0.5, 2.0, True, ("at least 3", "2")),
        (5, 0.5, 2.0, False, ()),
    ])
    def test_line_validation(self, numBullets: int, fastestTime: float, slowestTime: float, should_fail: bool, error_patterns: tuple[str, ...]):
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()
        comp.set_context(target=enums.EMPTY_EMITTER)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()
        
        # Setup caller
        caller = Component("Caller", 200)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                caller.instant.Line(
                    time=0, comp=comp, emitter=50, targetDir=90, bullet=lib.bullet2,
                    numBullets=numBullets, fastestTime=fastestTime, slowestTime=slowestTime, dist=100
                )
            assert_error(exc, *error_patterns)
        else:
            caller.instant.Line(
                time=0, comp=comp, emitter=50, targetDir=90, bullet=lib.bullet2,
                numBullets=numBullets, fastestTime=fastestTime, slowestTime=slowestTime, dist=100
            )
            assert len(caller.triggers) > 0


# ============================================================================
# TIMED PATTERNS - Validation
# ============================================================================

class TestTimedRadialWaveValidation:
    @pytest.mark.parametrize("waves,interval,should_fail,error_patterns", [
        (0, 0.5, True, ("waves must be at least 1",)),
        (1, 0.5, True, ("use instant.Radial",)),  # Single wave should use instant.Radial
        (3, -0.5, True, ("non-negative", "-0.5")),
        (3, 0.5, False, ()),
    ])
    def test_radial_wave_validation(self, spawn_ordered_comp: Component, waves: int, interval: float, should_fail: bool, error_patterns: tuple[str, ...]):
        # Setup caller with pointer circle
        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                caller.timed.RadialWave(
                    time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                    waves=waves, interval=interval, numBullets=12
                )
            assert_error(exc, *error_patterns)
        else:
            caller.timed.RadialWave(
                time=0, comp=spawn_ordered_comp, bullet=lib.bullet1,
                waves=waves, interval=interval, numBullets=12
            )
            assert len(caller.triggers) > 0


class TestTimedLineValidation:
    @pytest.mark.parametrize("numBullets,spacing,should_fail,error_patterns", [
        (1, 0.5, True, ("numBullets must be at least 2",)),
        (5, -0.1, True, ("non-negative", "-0.1")),
    ])
    def test_timed_line_validation(self, numBullets: int, spacing: float, should_fail: bool, error_patterns: tuple[str, ...]):
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        
        # Setup caller
        caller = Component("Caller", 200)
        
        if should_fail:
            with pytest.raises(ValueError) as exc:
                caller.timed.Line(time=0, comp=comp, targetDir=90, bullet=lib.bullet1,
                    numBullets=numBullets, spacing=spacing, t=1.0, dist=100)
            assert_error(exc, *error_patterns)
        else:
            caller.timed.Line(time=0, comp=comp, targetDir=90, bullet=lib.bullet1,
                numBullets=numBullets, spacing=spacing, t=1.0, dist=100)
            assert len(caller.triggers) > 0
    

# ============================================================================
# METHOD CHAINING - Return Self
# ============================================================================

class TestMethodChainingReturnsSelf:
    def test_chain_preserves_trigger_count(self, comp_with_target: Component):
        (comp_with_target
            .Toggle(0, activateGroup=True)
            .Alpha(0.1, opacity=50)
            .MoveBy(0.2, dx=10, dy=20)
        )
        assert len(comp_with_target.triggers) == 3


# ============================================================================
# COMPONENT TARGET ACCEPTS COMPONENT OBJECTS
# ============================================================================

class TestGeneralComponentFeatures:
    def test_component_without_spawn_order_warning(self, comp_with_target: Component):
        comp_with_target.Toggle(0, activateGroup=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lib.save_all(filename="testing")

        assert_warning(w, "spawn order", "Test")


# ============================================================================
# GET_TRIGGERS & HAS_TRIGGER_PROPERTIES
# ============================================================================

class TestGetTriggersMethod:
    """Test Component.get_triggers() and has_trigger_properties()"""

    def test_get_triggers_multiple_properties(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        comp.Pickup(0, item_id=12, count=100, override=False)

        result = comp.get_triggers({ppt.ITEM_ID: 12, ppt.PICKUP_COUNT: 50})
        assert len(result) == 1

    def test_get_triggers_wildcard_any(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=11, count=123, override=True)
        comp.PickupModify(0, item_id=11, factor=1.45, multiply=True)

        result = comp.get_triggers({ppt.PICKUP_MULTIPLY_DIVIDE: Any})
        assert len(result) == 2

    def test_get_triggers_by_object_id(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        comp.set_context(target=300)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        result = comp.get_triggers({ppt.OBJ_ID: enums.ObjectID.PICKUP})
        assert len(result) == 1

    def test_get_triggers_no_match(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        assert len(comp.get_triggers({ppt.ITEM_ID: 999})) == 0


class TestHasTriggerPropertiesMethod:
    def test_has_trigger_properties_multiple(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)

        assert comp.has_trigger_properties({ppt.ITEM_ID: 12, ppt.PICKUP_COUNT: 50}) is True
        assert comp.has_trigger_properties({ppt.ITEM_ID: 12, ppt.PICKUP_COUNT: 999}) is False

    def test_has_trigger_properties_wildcard(self):
        comp = Component("Test", 100, 5)
        comp.PickupModify(0, item_id=11, factor=1.45, multiply=True)

        assert comp.has_trigger_properties({ppt.PICKUP_MULTIPLY_DIVIDE: Any}) is True

    def test_has_trigger_properties_empty_dict_rejected(self):
        comp = Component("Test", 100, 5)
        with pytest.raises(ValueError) as exc_info:
            comp.has_trigger_properties({})
        assert_error(exc_info, "empty")

    def test_consistency_with_get_triggers(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)

        query = {ppt.ITEM_ID: 12}
        assert comp.has_trigger_properties(query) == (len(comp.get_triggers(query)) > 0)


# ============================================================================
# SPAWN LIMIT ENFORCEMENT (_enforce_spawn_limit)
# ============================================================================

class TestSpawnLimitEnforcement:
    def test_case1_two_unmapped_spawns_same_tick_with_spawn_trigger_rejected(self):
        """
        Case 1: Multiple unmapped spawns targeting same group with spawn trigger.

        This is the most common spawn limit bug - when you spawn the same component
        multiple times in the same tick, and that component contains spawn triggers.
        GD will only execute the first spawn, silently dropping the rest.
        """
        target_comp = Component("Target", 200)
        target_comp.assert_spawn_order(True)
        target_comp.Spawn(0, 300, spawnOrdered=True)  # Contains spawn trigger

        caller = Component("Caller", 100, 5)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True)  # Two spawns at X=0
        caller.Spawn(0, 200, spawnOrdered=True)  # targeting same group

        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([caller, target_comp])

        assert_error(exc_info, "spawn limit violation", "case 1", "unmapped",
                     "2 simultaneous", "group 200")

    def test_case1_different_ticks_allowed(self):
        """
        Spawns at different X coordinates (different ticks) should be allowed.
        This tests that spawn_order grouping works correctly.
        """
        target_comp = Component("Target", 200)
        target_comp.assert_spawn_order(True)
        target_comp.Spawn(0, 300, spawnOrdered=True)

        caller = Component("Caller", 100)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True)    # X=0
        caller.Spawn(0.5, 200, spawnOrdered=True)  # X=0.5 (different tick)

        # Should not raise
        lib._enforce_spawn_limit([caller, target_comp])

    def test_case1_no_spawn_order_treats_all_as_same_tick(self):
        """
        Without spawn_order, all triggers are considered same tick.
        This tests the requireSpawnOrder=False path.
        """
        target_comp = Component("Target", 200)
        target_comp.assert_spawn_order(True)
        target_comp.Spawn(0, 300, spawnOrdered=True)

        caller = Component("Caller", 100)

        caller.assert_spawn_order(False) # Explicitly set to False to test that path
        caller.Spawn(0, 200, spawnOrdered=True)
        caller.Spawn(99, 200, spawnOrdered=True)  # Different X, but no spawn_order

        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([caller, target_comp])

        assert_error(exc_info, "case 1")

    def test_case1_different_targets_allowed(self):
        """Multiple spawns to different groups should be allowed."""
        target1 = Component("Target1", 200)
        target1.assert_spawn_order(True)
        target1.Spawn(0, 300, spawnOrdered=True)

        target2 = Component("Target2", 201)
        target2.assert_spawn_order(True)
        target2.Spawn(0, 301, spawnOrdered=True)

        caller = Component("Caller", 100)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True)  # Different targets
        caller.Spawn(0, 201, spawnOrdered=True)

        # Should not raise
        lib._enforce_spawn_limit([caller, target1, target2])

    def test_case1_spawn_delay_excludes_from_check(self):
        """Spawns with delay > 0 should not be checked (they execute at different times)."""
        target_comp = Component("Target", 200)
        target_comp.assert_spawn_order(True)
        target_comp.Spawn(0, 300, spawnOrdered=True)

        caller = Component("Caller", 100)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True, delay=0)    # Immediate
        caller.Spawn(0, 200, spawnOrdered=True, delay=0.1)  # Delayed, excluded

        # Should not raise (only 1 immediate spawn)
        lib._enforce_spawn_limit([caller, target_comp])

    def test_case1_target_without_spawn_allowed(self):
        """Multiple spawns targeting group without spawn triggers should be allowed."""
        target_comp = Component("Target", 200)
        target_comp.assert_spawn_order(True)
        target_comp.set_context(target=300)
        target_comp.Toggle(0, activateGroup=True)  # Not a spawn trigger
        target_comp.clear_context()

        caller = Component("Caller", 100)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True)
        caller.Spawn(0, 200, spawnOrdered=True)

        # Should not raise (target has no spawn triggers)
        lib._enforce_spawn_limit([caller, target_comp])

    def test_case2_remapped_spawn_to_multiple_spawns_rejected(self):
        """
        Case 2: A has remapped spawn, B has multiple simultaneous triggers to C.

        When A has a remap, and B has 2+ simultaneous triggers targeting C (which has spawns),
        C gets spawn limited unless B's triggers have reset_remap.

        B's triggers must have remaps themselves (otherwise Case 1 catches it).
        """
        # Layer C: Final target that contains spawn triggers
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        # Layer B: Component with multiple spawns to same target (C)
        # Both have remaps so Case 1 doesn't apply (only 1 unmapped allowed)
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400")  # Has remap
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400")  # Has remap

        # Layer A: Has remapped spawn trigger to B
        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True, remap="997.200")

        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

        assert_error(exc_info, "spawn limit violation", "case 2", "remap")

    def test_case2_single_spawn_in_b_allowed(self):
        """Case 2 should not trigger if B only has 1 spawn trigger."""
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True)  # Only 1 spawn

        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True, remap="999.200")

        # Should not raise (B has only 1 simultaneous spawn)
        lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

    def test_case2_no_remap_in_a_allowed(self):
        """Case 2 should not trigger if A has no remap."""
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        # B has 2 simultaneous spawns but they have remaps
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400")
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400")

        # A has NO remap - so Case 2 shouldn't apply
        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True)  # No remap

        # Should not raise (A has no remap, and B's triggers are remapped so Case 1 doesn't apply either)
        lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

    def test_case1_c_has_reset_remap_treats_all_as_unmapped(self):
        """
        Case 1 special: If C has reset_remap, ALL of B's triggers are treated as unmapped.

        Even if B's triggers have remaps, C's reset_remap makes them act unmapped.
        """
        # C has a spawn trigger with reset_remap=True
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True, reset_remap=True)

        # B has 2 simultaneous spawns WITH remaps (normally Case 1 wouldn't apply)
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400")
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400")

        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([layer_b, layer_c])

        assert_error(exc_info, "case 1", "reset_remap")

    def test_case2_b_has_reset_remap_escapes_violation(self):
        """
        Case 2 escape: If B's triggers have reset_remap, they ignore A's remap.

        We can tolerate 1 trigger without reset_remap (limiting 1 to 1 is fine).
        """
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        # B has 2 spawns, but BOTH have reset_remap - they ignore A's remap
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400", reset_remap=True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400", reset_remap=True)

        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True, remap="997.200")

        # Should not raise - all B triggers have reset_remap
        lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

    def test_case2_b_has_one_without_reset_remap_allowed(self):
        """
        Case 2 escape: We tolerate exactly 1 trigger without reset_remap.

        Limiting 1 to 1 is the same as not limiting at all.
        """
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        # B has 2 spawns, only 1 has reset_remap
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400", reset_remap=True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400")

        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True, remap="997.200")

        # Should not raise - only 1 trigger lacks reset_remap
        lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

    def test_case2_b_has_two_without_reset_remap_rejected(self):
        """
        Case 2: If 2+ triggers in B don't have reset_remap, violation occurs.
        """
        layer_c = Component("LayerC", 400)
        layer_c.assert_spawn_order(True)
        layer_c.Spawn(0, 500, spawnOrdered=True)

        # B has 3 spawns, only 1 has reset_remap (2 without = violation)
        layer_b = Component("LayerB", 200)
        layer_b.assert_spawn_order(True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="999.400", reset_remap=True)
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="998.400")
        layer_b.Spawn(0, 400, spawnOrdered=True, remap="997.400")

        layer_a = Component("LayerA", 100)
        layer_a.assert_spawn_order(True)
        layer_a.Spawn(0, 200, spawnOrdered=True, remap="996.200")

        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([layer_a, layer_b, layer_c])

        assert_error(exc_info, "case 2", "reset_remap")

    def test_integration_with_save_all_default_enabled(self):
        """Test that save_all() calls _enforce_spawn_limit by default"""
        target = Component("Target", 200)
        target.assert_spawn_order(True)
        target.Spawn(0, 300, spawnOrdered=True)

        caller = Component("Caller", 100)
        caller.assert_spawn_order(True)
        caller.Spawn(0, 200, spawnOrdered=True)
        caller.Spawn(0, 200, spawnOrdered=True)

        with pytest.raises(RuntimeError) as exc_info:
            lib.save_all(filename="testing")

        assert_error(exc_info, "spawn limit")

    def test_multitarget_with_spawn_triggers_warns(self):
        """
        Multitarget should warn if the component contains spawn triggers.

        Multitarget creates binary spawn trees, which would multiply
        spawn triggers exponentially, causing spawn limit violations.
        """
        comp = Component("WithSpawn", 100)
        comp.Spawn(0, 200, spawnOrdered=True)
        comp.set_context(target=300)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            Multitarget._get_binary_components(3, comp)

        assert_warning(w, "spawn limit", "WithSpawn", "multitarget", "spawn triggers")


# ============================================================================
# TOLERANCE BOUNDARY TESTS (Validates correct grouping behavior)
# ============================================================================

class TestToleranceBoundaryEdgeCases:
    def test_triggers_beyond_tolerance_are_separated(self):
        """
        Triggers beyond tolerance should be in separate groups (no violation).

        Setup: B has 3 spawns at time 0.0s, ~0.004s, ~0.008s
        Expected: Only first two are grouped (~0.004s converts to ~1.25 studs, within ~1.3 tolerance)
            Third is separate (~0.008s converts to ~2.49 studs, beyond tolerance)
            Only 2 simultaneous spawns → violation
        """
        target = Component("Target", 300)
        target.assert_spawn_order(True)
        target.Spawn(0, 400, spawnOrdered=True)

        caller = Component("Caller", 200)
        caller.assert_spawn_order(True)
        caller.Spawn(0.0, 300, spawnOrdered=True)
        caller.Spawn(0.004, 300, spawnOrdered=True)  # ~1.25 studs, within tolerance
        caller.Spawn(0.008, 300, spawnOrdered=True)  # ~2.49 studs, beyond tolerance

        # Should raise - first two are grouped together (2 simultaneous)
        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([caller, target])

        assert_error(exc_info, "case 1", "2 simultaneous")

    def test_triggers_within_tolerance_are_grouped(self):
        """
        Triggers within tolerance should be grouped together (violation).

        Setup: B has 2 spawns at time 0.0s and ~0.004s (converts to ~1.25 studs, within ~1.3 tolerance)
        Expected: Both grouped together → 2 simultaneous → violation
        """
        target = Component("Target", 300)
        target.assert_spawn_order(True)
        target.Spawn(0, 400, spawnOrdered=True)

        caller = Component("Caller", 200)
        caller.assert_spawn_order(True)
        caller.Spawn(0.0, 300, spawnOrdered=True)
        caller.Spawn(0.004, 300, spawnOrdered=True)  # ~1.25 studs, within tolerance

        # Should raise - both grouped together
        with pytest.raises(RuntimeError) as exc_info:
            lib._enforce_spawn_limit([caller, target])

        assert_error(exc_info, "case 1", "2 simultaneous")


# ============================================================================
# SOLID GROUP ENFORCEMENT
# ============================================================================

class TestSolidGroupEnforcement:
    def test_solid_group_as_component_caller_rejected(self):
        """A solid group used as a component caller should raise ValueError"""
        comp = Component("BadComponent", callerGroup=100)
        lib.all_components.append(comp)
        
        with pytest.raises(ValueError) as exc:
            lib._validate_solid_groups(100)
        
        assert_error(exc, "100", "component caller", "BadComponent")

    def test_solid_group_in_context_groups_rejected(self):
        """A solid group added via set_context(groups=...) should raise ValueError"""
        comp = Component("BadComponent", callerGroup=50)
        comp.set_context(groups=100)
        lib.all_components.append(comp)
        
        with pytest.raises(ValueError) as exc:
            lib._validate_solid_groups(100)
        
        assert_error(exc, "100", "component caller")

    def test_complex_scene_conflict_detected(self):
        """Complex scene with solid group used as both target and caller"""
        bullet = Component("Bullet", callerGroup=500)
        bullet.set_context(target=600)
        bullet.MoveTowards(0, targetDir=1000, t=1.0, dist=100)
        lib.all_components.append(bullet)
        
        emitter = Component("Emitter", callerGroup=501)
        emitter.set_context(target=700)
        emitter.GotoGroup(0, location=2000, t=1.0)
        lib.all_components.append(emitter)
        
        # Bad component - uses 600 as caller (which is also bullet's solid target)
        bad = Component("BadPattern", callerGroup=600)
        bad.set_context(target=999)
        bad.Toggle(0, activateGroup=True)
        lib.all_components.append(bad)
        
        with pytest.raises(ValueError) as exc:
            lib._validate_solid_groups()
        
        assert_error(exc, "600", "BadPattern")

    def test_valid_scene_with_separate_groups(self):
        bullet = Component("Bullet", callerGroup=500)
        bullet.set_context(target=600)
        bullet.MoveTowards(0, targetDir=1000, t=1.0, dist=100)
        lib.all_components.append(bullet)
        
        pattern = Component("Pattern", callerGroup=501)
        pattern.Spawn(0, target=2000, spawnOrdered=False)
        lib.all_components.append(pattern)
        
        # Should not raise - groups are properly separated
        lib._validate_solid_groups()


# ============================================================================
# BEZIER MOVE - Movement Accuracy Tests
# ============================================================================

class TestBezierMove:
    """
    Test that BezierMove generates correct total displacement.
    
    X-axis movement should be exact (uses simple ease-in/ease-out).
    Y-axis has small errors from polynomial approximation (< 2.6% or 0.5 units).
    """
    
    def _get_bezier_displacement(self, curve_label: CurveType, dx: float, dy: float, duration: float):
        """
        Helper to set up a BezierMove and return total displacement.
        Returns tuple of (total_dx, total_dy).
        """
        comp = Component("BezierTest", 100)
        comp.set_context(target=500)
        
        comp.timed.BezierMove(
            time=0,
            curve_label=curve_label,
            dx=dx,
            dy=dy,
            t=duration
        )
        
        move_triggers = [t for t in comp.triggers if t.get(ppt.OBJ_ID) == enums.ObjectID.MOVE]
        total_dx = float(sum(t.get(ppt.MOVE_X, 0) for t in move_triggers))
        total_dy = float(sum(t.get(ppt.MOVE_Y, 0) for t in move_triggers))
        
        return total_dx, total_dy
    
    def _check_x_exact(self, expected: float, actual: float):
        """Check X-axis movement is exact."""
        assert abs(actual - expected) < 1e-6, \
            f"X-axis: Expected {expected}, got {actual}"
    
    def _check_y_tolerance(self, expected: float, actual: float, curve_name: str = ""):
        """Check Y-axis movement is within tolerance (2.6% or 0.5 units)."""
        tolerance = max(abs(expected) * 0.026, 0.5)  # 2.6% or 0.5 units
        error = abs(actual - expected)
        
        curve_info = f" (curve: {curve_name})" if curve_name else ""
        assert error < tolerance, \
            f"Y-axis{curve_info}: Expected {expected}, got {actual}, error={error:.6f}, tolerance={tolerance}"
    
    @pytest.mark.parametrize("curve_type,dx,dy,duration", [
        ("GENTLE_ARC", 100.0, 50.0, 2.0),
        ("S_CURVE", -75.0, -120.0, 1.5),
        ("SMOOTH_EASE", 0.0, 0.0, 1.0),
        ("FAST_ARC", 200.0, 0.0, 1.0, ),
        ("STEEP_RISE", 0.0, 150.0, 2.0, ),
        ("GENTLE_ARC", 1000.0, 2000.0, 3.0),
        ("SMOOTH_EASE", 0.5, 0.25, 0.5),
    ])
    def test_bezier_move_displacement(self, curve_type: str, dx: float, dy: float, duration: float):
        """Test BezierMove displacement accuracy across various scenarios."""
        
        total_dx, total_dy = self._get_bezier_displacement(
            getattr(CurveType, curve_type), dx, dy, duration
        )
        
        self._check_x_exact(dx, total_dx)
        
        # For zero/near-zero movement, use exact check instead of percentage tolerance
        if abs(dy) < 1e-6:
            assert abs(total_dy) < 1e-6
        else:
            self._check_y_tolerance(dy, total_dy, curve_type)
    
    def test_bezier_move_horizontal_only(self):
        """Test movement with dy=0 (horizontal only)."""
        total_dx, total_dy = self._get_bezier_displacement(
            CurveType.GENTLE_ARC, dx=100, dy=0, duration=1.0
        )
        self._check_x_exact(100, total_dx)
        assert abs(total_dy) < 1e-6
    
    def test_bezier_move_vertical_only(self):
        """Test movement with dx=0 (vertical only)."""
        total_dx, total_dy = self._get_bezier_displacement(
            CurveType.STEEP_RISE, dx=0, dy=150, duration=1.0
        )
        self._check_x_exact(0, total_dx)
        self._check_y_tolerance(150, total_dy, "STEEP_RISE")
    
    def test_bezier_move_very_short_duration(self):
        """Test with minimal duration."""
        total_dx, total_dy = self._get_bezier_displacement(
            CurveType.SMOOTH_EASE, dx=10, dy=10, duration=0.01
        )
        self._check_x_exact(10, total_dx)
        self._check_y_tolerance(10, total_dy)
    
    @pytest.mark.parametrize("curve_type", [
        "BOSS_CHARGE",
        "BOSS_WEAVE", 
        "FAST_ARC",
        "GENTLE_ARC",
        "GENTLE_ARC_DOWN",
        "S_CURVE",
        "S_CURVE_REVERSE",
        "SMOOTH_EASE",
        "STEEP_DIVE",
        "STEEP_RISE",
    ])
    def test_bezier_move_all_curve_types(self, curve_type: str):
        """Test displacement accuracy across all registered curve types."""
        
        dx_requested = 80.0
        dy_requested = 120.0
        
        total_dx, total_dy = self._get_bezier_displacement(
            getattr(CurveType, curve_type), dx_requested, dy_requested, duration=1.5
        )
        
        self._check_x_exact(dx_requested, total_dx)
        self._check_y_tolerance(dy_requested, total_dy, curve_type)
    
    
    @pytest.mark.parametrize("set_target,time,duration,curve,exception_type,error_patterns", [
        (False, 0, 1.0, "GENTLE_ARC", ValueError, ("no target", "set_context")),
        (True, 0, -1.0, "GENTLE_ARC", ValueError, ("non-negative", "-1")),
        (True, -0.5, 1.0, "GENTLE_ARC", ValueError, ("non-negative", "-0.5")),
        (True, 0, 1.0, "nonexistent_curve", KeyError, ("not registered", "nonexistent_curve")),
    ])
    def test_bezier_move_validation_errors(self, set_target: bool, time: float, duration: float,
                                          curve: str, exception_type: Exception, error_patterns: tuple[str,...]):
        """Test that BezierMove validates parameters and rejects invalid inputs."""
        
        comp = Component("BezierTest", 100)
        if set_target:
            comp.set_context(target=500)
        
        # Get curve label - use string directly if not a valid CurveType
        if hasattr(CurveType, curve):
            curve_label = getattr(CurveType, curve)
        else:
            curve_label = curve
        
        with pytest.raises(exception_type) as exc:
            comp.timed.BezierMove(
                time=time,
                curve_label=curve_label,
                dx=100,
                dy=50,
                t=duration
            )
        
        assert_error(exc, *error_patterns)
    
    @pytest.mark.parametrize("invalid_curve", [None, ""])
    def test_bezier_move_invalid_curve_type(self, invalid_curve: CurveType):
        """Test that None and empty string curve labels are rejected."""
        comp = Component("BezierTest", 100)
        comp.set_context(target=500)
        
        with pytest.raises((KeyError, TypeError, AttributeError)):
            comp.timed.BezierMove(
                time=0, curve_label=invalid_curve,
                dx=100, dy=50, t=1.0
            )
    
    def test_bezier_move_generates_multiple_triggers(self):
        """Test that BezierMove generates multiple MoveBy triggers."""
        
        comp = Component("BezierTest", 100)
        comp.set_context(target=500)
        
        comp.timed.BezierMove(time=0, curve_label=CurveType.GENTLE_ARC, dx=100, dy=50, t=1.0)
        
        move_triggers = [t for t in comp.triggers if t.get(ppt.OBJ_ID) == enums.ObjectID.MOVE]
        assert len(move_triggers) >= 2, "BezierMove should generate at least 2 MoveBy triggers"
