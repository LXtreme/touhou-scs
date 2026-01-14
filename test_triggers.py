
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

def setup_pointer_circle(caller: Component) -> Component:
    """Helper to set up a PointerCircle context for pattern tests."""
    caller.assert_spawn_order(True)
    caller.pointer.SetPointerCircle(0, location=100, follow=False)
    return caller

P = enums.Properties

def assert_error(exc_info: ExceptionInfo[BaseException], *patterns: str) -> None:
    """Assert exception message contains all patterns."""
    msg_clean = str(exc_info.value).lower().replace(" ", "")
    for pattern in patterns:
        p = pattern.lower().replace(" ", "")
        assert p in msg_clean, f"Expected '{pattern}' in: {str(exc_info.value)}"

def assert_warning(warning_list: list[warnings.WarningMessage], *patterns: str) -> None:
    """Assert warning list contains all patterns."""
    combined_msgs = " ".join(str(w.message).lower().replace(" ", "") for w in warning_list)
    for pattern in patterns:
        p = pattern.lower().replace(" ", "")
        assert p in combined_msgs, f"Expected '{pattern}' in warnings."

# ============================================================================
# SPAWN TRIGGER - Target Group Validation
# ============================================================================

class TestSpawnTargetValidation:
    """Test Spawn target group validation boundaries"""

    def test_spawn_target_zero_rejected(self):
        """Target group 0 is out of valid range"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 0, spawnOrdered=False)
        assert_error(exc, "positive", "0")

    def test_spawn_target_negative_rejected(self):
        """Negative target groups are rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, -50, spawnOrdered=False)
        assert_error(exc, "positive", "-50")

    def test_spawn_target_10_valid(self):
        """Target group 10 is valid (non-restricted)"""
        comp = Component("Test", 100)
        comp.Spawn(0, 10, spawnOrdered=False)
        trigger = comp.triggers[0]
        assert trigger[P.TARGET] == 10

    def test_spawn_target_at_counter_valid(self):
        """Target at unknown_g.counter boundary is valid"""
        comp = Component("Test", 100)
        max_valid = utils.unknown_g.counter
        comp.Spawn(0, max_valid, spawnOrdered=False)
        trigger = comp.triggers[0]
        assert trigger[P.TARGET] == max_valid

    def test_spawn_target_above_counter_rejected(self):
        """Target above unknown_g.counter is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, utils.unknown_g.counter + 1, spawnOrdered=False)
        assert_error(exc, "out of valid range")

    def test_spawn_restricted_group_rejected(self):
        """Restricted groups are rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 3, spawnOrdered=False)
        assert_error(exc, "restricted", "3")


class TestSpawnDelayValidation:
    """Test Spawn delay parameter behavior"""

    def test_spawn_negative_delay_rejected(self):
        """Negative delay is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 50, spawnOrdered=False, delay=-1)
        assert_error(exc, "non-negative", "-1")

    def test_spawn_zero_delay_not_stored(self):
        """Zero delay is not stored"""
        comp = Component("Test", 100)
        comp.Spawn(0, 50, spawnOrdered=False, delay=0)
        trigger = comp.triggers[0]
        assert P.SPAWN_DELAY not in trigger

    def test_spawn_positive_delay_stored(self):
        """Positive delay is stored"""
        comp = Component("Test", 100)
        comp.Spawn(0, 50, spawnOrdered=False, delay=0.5)
        trigger = comp.triggers[0]
        assert trigger[P.SPAWN_DELAY] == 0.5


class TestSpawnRemapValidation:
    """Test remap string handling"""

    def test_spawn_remap_empty_string_not_stored(self):
        """Empty remap string is silently skipped"""
        comp = Component("Test", 100)
        comp.Spawn(0, 50, spawnOrdered=False, remap="")
        trigger = comp.triggers[0]
        assert P.REMAP_STRING not in trigger

    def test_spawn_remap_odd_pairs_rejected(self):
        """Odd number of remap values is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 50, spawnOrdered=False, remap="1.2.3")
        assert_error(exc, "even number")

    def test_spawn_remap_duplicate_source_rejected(self):
        """Remapping same source to different targets is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Spawn(0, 50, spawnOrdered=False, remap="10.20.10.30")
        assert_error(exc, "Duplicate source", "10")


# ============================================================================
# MOVE TRIGGERS - Easing Boundaries
# ============================================================================

class TestMoveEasingValidation:
    """Test Move trigger easing parameter boundaries"""

    def test_easing_type_negative_rejected(self):
        """Negative easing type is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, type=-1)
        assert_error(exc, "type", "-1")

    def test_easing_type_19_rejected(self):
        """Easing type 19 (above max 18) is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, type=19)
        assert_error(exc, "type", "19")

    def test_easing_type_0_valid(self):
        """Easing type 0 (NONE) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, type=0)
        trigger = comp.triggers[0]
        assert trigger[P.EASING] == 0

    def test_easing_type_18_valid(self):
        """Easing type 18 (max) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, type=18)
        trigger = comp.triggers[0]
        assert trigger[P.EASING] == 18

    def test_easing_rate_0_10_rejected(self):
        """Easing rate at 0.10 is rejected (must be > 0.10)"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, rate=0.10)
        assert_error(exc, "rate", "0.1")

    def test_easing_rate_below_0_10_rejected(self):
        """Easing rate below 0.10 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, rate=0.05)
        assert_error(exc, "rate", "0.05")

    def test_easing_rate_above_20_rejected(self):
        """Easing rate above 20.0 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, rate=20.01)
        assert_error(exc, "rate", "20.01")

    def test_easing_rate_20_valid(self):
        """Easing rate 20.0 is valid (upper boundary)"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, rate=20.0)
        trigger = comp.triggers[0]
        assert trigger[P.EASING_RATE] == 20.0

    def test_easing_rate_just_above_0_10_valid(self):
        """Easing rate just above 0.10 is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, rate=0.11)
        trigger = comp.triggers[0]
        assert trigger[P.EASING_RATE] == 0.11

    def test_easing_type_float_non_integer_rejected(self):
        """Easing type as non-integer float is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=1.0, dist=100, type=2.5)
        assert_error(exc, "type", "2.5")


class TestMoveDurationValidation:
    """Test Move trigger duration validation"""

    def test_duration_negative_rejected(self):
        """Negative duration is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.MoveTowards(0, targetDir=60, t=-0.5, dist=100)
        assert_error(exc, "non-negative", "-0.5")

    def test_duration_zero_sets_silent(self):
        """Duration zero sets MOVE_SILENT flag"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=0, dist=100)
        trigger = comp.triggers[0]
        assert trigger[P.MOVE_SILENT] is True

    def test_duration_positive_no_silent(self):
        """Positive duration doesn't set MOVE_SILENT flag"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.MoveTowards(0, targetDir=60, t=0.5, dist=100)
        trigger = comp.triggers[0]
        assert P.MOVE_SILENT not in trigger


# ============================================================================
# ALPHA TRIGGER - Opacity Boundaries
# ============================================================================

class TestAlphaOpacityValidation:
    """Test Alpha trigger opacity boundaries"""

    def test_opacity_negative_rejected(self):
        """Negative opacity is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Alpha(0, opacity=-1)
        assert_error(exc, "between 0 and 100")

    def test_opacity_above_100_rejected(self):
        """Opacity above 100 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Alpha(0, opacity=101)
        assert_error(exc, "between 0 and 100")

    def test_opacity_0_valid(self):
        """Opacity 0 (fully transparent) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Alpha(0, opacity=0)
        trigger = comp.triggers[0]
        assert trigger[P.OPACITY] == 0.0

    def test_opacity_100_valid(self):
        """Opacity 100 (fully opaque) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Alpha(0, opacity=100)
        trigger = comp.triggers[0]
        assert trigger[P.OPACITY] == 1.0

    def test_opacity_converts_to_decimal(self):
        """Opacity 50 converts to 0.5"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Alpha(0, opacity=50)
        trigger = comp.triggers[0]
        assert trigger[P.OPACITY] == 0.5


# ============================================================================
# SCALE TRIGGER - Factor Validation
# ============================================================================

class TestScaleFactorValidation:
    def test_scale_factor_zero_rejected(self):
        """Scale factor 0 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Scale(0, factor=0, t=1.0)
        assert_error(exc, "factor", ">0", "0")

    def test_scale_factor_negative_rejected(self):
        """Negative scale factor is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Scale(0, factor=-1.0, t=1.0)
        assert_error(exc, "factor", ">0", "-1")

    def test_scale_factor_one_rejected(self):
        """Scale factor 1.0 (no change) is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Scale(0, factor=1.0, t=1.0)
        assert_error(exc, "1", "has no effect")

    def test_scale_factor_barely_above_one_valid(self):
        """Scale factor just above 1.0 is valid"""
        comp = Component("Test", 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comp.set_context(target=50)
            comp.Scale(0, factor=1.0001, t=1.0)
            assert_warning(w, "hold", "0", "not in reverse")
        assert len(comp.triggers) == 1

    def test_scale_factor_barely_below_one_valid(self):
        """Scale factor just below 1.0 is valid"""
        comp = Component("Test", 100)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comp.set_context(target=50)
            comp.Scale(0, factor=0.9999, t=1.0)
            assert_warning(w, "hold", "0", "not in reverse")
        assert len(comp.triggers) == 1

    def test_scale_hold_negative_rejected(self):
        """Negative hold time is rejected via duration validation"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Scale(0, factor=2.0, t=1.0, hold=-0.1)
        assert_error(exc, "non-negative", "-0.1")


# ============================================================================
# COUNT TRIGGER - Item ID Validation
# ============================================================================

class TestCountItemIdValidation:
    """Test Count trigger item ID validation"""

    def test_count_item_id_zero_rejected(self):
        """Item ID 0 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Count(0, item_id=0, count=5, activateGroup=True)
        assert_error(exc, "positive", "0")

    def test_count_item_id_negative_rejected(self):
        """Negative item ID is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Count(0, item_id=-1, count=5, activateGroup=True)
        assert_error(exc, "positive", "-1")

    def test_count_item_id_above_9999_rejected(self):
        """Item ID above 9999 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Count(0, item_id=10000, count=5, activateGroup=True)
        assert_error(exc, "positive", "10000")

    def test_count_item_id_1_valid(self):
        """Item ID 1 (lower boundary) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Count(0, item_id=1, count=5, activateGroup=True)
        trigger = comp.triggers[0]
        assert trigger[P.ITEM_ID] == 1

    def test_count_item_id_9999_valid(self):
        """Item ID 9999 (upper boundary) is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Count(0, item_id=9999, count=5, activateGroup=True)
        trigger = comp.triggers[0]
        assert trigger[P.ITEM_ID] == 9999


# ============================================================================
# PICKUP TRIGGER - Validation
# ============================================================================

class TestPickupValidation:
    """Test Pickup trigger validation"""

    def test_pickup_item_id_zero_rejected(self):
        """Item ID 0 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Pickup(0, item_id=0, count=50, override=False)
        assert_error(exc, "positive", "0")

    def test_pickup_item_id_negative_rejected(self):
        """Negative item ID is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Pickup(0, item_id=-10, count=50, override=False)
        assert_error(exc, "positive", "-10")

    def test_pickup_item_id_above_9999_rejected(self):
        """Item ID above 9999 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Pickup(0, item_id=10000, count=50, override=False)
        assert_error(exc, "positive", "10000")

    def test_pickup_count_zero_rejected(self):
        """Count of 0 (no change) is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.Pickup(0, item_id=5, count=0, override=False)
        assert_error(exc, "no change", "0")

    def test_pickup_count_negative_allowed(self):
        """Negative count (subtract items) is allowed"""
        comp = Component("Test", 100)
        comp.Pickup(0, item_id=5, count=-50, override=False)
        trigger = comp.triggers[0]
        assert trigger[P.PICKUP_COUNT] == -50

    def test_pickup_no_target_property(self):
        """Pickup trigger should not have TARGET property"""
        comp = Component("Test", 100)
        comp.Pickup(0, item_id=5, count=50, override=False)
        trigger = comp.triggers[0]
        assert P.TARGET not in trigger


# ============================================================================
# PICKUP MODIFY TRIGGER - Validation
# ============================================================================

class TestPickupModifyValidation:
    """Test PickupModify trigger validation"""

    def test_pickup_modify_item_id_zero_rejected(self):
        """Item ID 0 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=0, factor=1.5, multiply=True)
        assert_error(exc, "positive", "0")

    def test_pickup_modify_item_id_above_9999_rejected(self):
        """Item ID above 9999 is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=10000, factor=1.5, multiply=True)
        assert_error(exc, "positive", "10000")

    def test_pickup_modify_factor_one_rejected(self):
        """Factor of 1 (no effect) is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=5, factor=1, multiply=True)
        assert_error(exc, "1 has no effect")

    def test_pickup_modify_factor_one_float_rejected(self):
        """Factor of 1.0 (no effect) is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=5, factor=1.0, multiply=True)
        assert_error(exc, "1 has no effect")

    def test_pickup_modify_no_mode_rejected(self):
        """Neither multiply nor divide specified is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=5, factor=1.5)
        assert_error(exc, "multiply", "divide")

    def test_pickup_modify_both_modes_rejected(self):
        """Both multiply and divide specified is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.PickupModify(0, item_id=5, factor=1.5, multiply=True, divide=True)
        assert_error(exc, "both", "multiply", "divide")

    def test_pickup_modify_multiply_mode_value(self):
        """Multiply mode sets PICKUP_MULTIPLY_DIVIDE to 1"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=5, factor=1.5, multiply=True)
        trigger = comp.triggers[0]
        assert trigger[P.PICKUP_MULTIPLY_DIVIDE] == 1

    def test_pickup_modify_divide_mode_value(self):
        """Divide mode sets PICKUP_MULTIPLY_DIVIDE to 2"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=5, factor=2.0, divide=True)
        trigger = comp.triggers[0]
        assert trigger[P.PICKUP_MULTIPLY_DIVIDE] == 2

    def test_pickup_modify_no_target_property(self):
        """PickupModify trigger should not have TARGET property"""
        comp = Component("Test", 100)
        comp.PickupModify(0, item_id=5, factor=1.5, multiply=True)
        trigger = comp.triggers[0]
        assert P.TARGET not in trigger


# ============================================================================
# ROTATE / POINT TO GROUP - Dynamic + Easing
# ============================================================================

class TestPointToGroupValidation:
    """Test PointToGroup dynamic + easing conflict"""

    def test_point_to_group_dynamic_with_easing_type_rejected(self):
        """Dynamic mode with easing type is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.PointToGroup(0, targetDir=60, t=1.0, dynamic=True, type=2)
        assert_error(exc, "dynamic", "easing", "2")

    def test_point_to_group_dynamic_with_easing_rate_rejected(self):
        """Dynamic mode with non-default easing rate is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.PointToGroup(0, targetDir=60, t=1.0, dynamic=True, rate=2.0)
        assert_error(exc, "dynamic", "easing", "2.0")

    def test_point_to_group_dynamic_without_easing_valid(self):
        """Dynamic mode without easing is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.PointToGroup(0, targetDir=60, t=1.0, dynamic=True)
        trigger = comp.triggers[0]
        assert trigger[P.DYNAMIC] is True

    def test_point_to_group_static_with_easing_valid(self):
        """Static mode with easing is valid"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.PointToGroup(0, targetDir=60, t=1.0, type=2, rate=1.5)
        trigger = comp.triggers[0]
        assert trigger[P.EASING] == 2
        assert trigger[P.EASING_RATE] == 1.5


class TestRotateValidation:
    """Test Rotate trigger validation"""

    def test_rotate_center_defaults_to_target(self):
        """Center defaults to target when not specified"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Rotate(0, angle=45, t=1.0)
        trigger = comp.triggers[0]
        assert trigger[P.ROTATE_CENTER] == 50

    def test_rotate_center_can_differ_from_target(self):
        """Center can be different from target"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Rotate(0, angle=45, center=60, t=1.0)
        trigger = comp.triggers[0]
        assert trigger[P.TARGET] == 50
        assert trigger[P.ROTATE_CENTER] == 60

    def test_rotate_restricted_center_rejected(self):
        """Restricted center group is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context(target=50)
            comp.Rotate(0, angle=45, center=1, t=1.0)
        assert_error(exc, "restricted", "1")


# ============================================================================
# GROUP CONTEXT - State Management
# ============================================================================

class TestGroupContextManagement:
    """Test group context state management - could easily corrupt state"""

    def test_start_context_adds_groups_to_subsequent_triggers(self):
        """Triggers after set_context include context groups"""
        comp = Component("Test", 100)
        comp.set_context(groups=200)
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)
        trigger = comp.triggers[0]
        assert 200 in trigger[P.GROUPS]

    def test_end_context_removes_context_groups(self):
        """Triggers after clear_context exclude context groups"""
        comp = Component("Test", 100)
        comp.set_context(groups=200)
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context(groups_only=True)
        comp.set_context(target=51)
        comp.Toggle(0.1, activateGroup=True)

        assert 200 in comp.triggers[0][P.GROUPS]
        assert 200 not in comp.triggers[1][P.GROUPS]

    def test_nested_context_overwrites(self):
        """Setting groups context while one is active overwrites it"""
        comp = Component("Test", 100)
        comp.set_context(groups=200)
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(groups=300)  # Overwrites previous group context
        comp.set_context(target=51)
        comp.Toggle(0.1, activateGroup=True)

        assert 200 in comp.triggers[0][P.GROUPS]
        assert 200 not in comp.triggers[1][P.GROUPS]
        assert 300 in comp.triggers[1][P.GROUPS]

    def test_clear_context_is_idempotent(self):
        """Clearing context without setting is safe (idempotent)"""
        comp = Component("Test", 100)
        # Should not raise - clearing empty context is safe
        comp.clear_context(groups_only=True)
        comp.clear_context()
        assert comp.groups == [comp.caller]
        assert comp.target == -1

    def test_context_with_multiple_groups(self):
        """Context can add multiple groups at once"""
        comp = Component("Test", 100)
        comp.set_context(groups=[200, 201, 202])
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)
        trigger = comp.triggers[0]
        assert all(g in trigger[P.GROUPS] for g in [200, 201, 202])

    def test_context_with_list_groups(self):
        """Context accepts list of groups"""
        comp = Component("Test", 100)
        comp.set_context(groups=[200, 201])
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)
        trigger = comp.triggers[0]
        assert 200 in trigger[P.GROUPS]
        assert 201 in trigger[P.GROUPS]

    def test_context_empty_rejected(self):
        """Setting context with no parameters is rejected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError) as exc:
            comp.set_context()
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
        assert comp.triggers[0][P.TARGET] == 50
        assert comp.triggers[1][P.TARGET] == 60

    def test_temp_context_restores_groups(self):
        """temp_context restores groups after exit"""
        comp = Component("Test", 100)
        comp.set_context(target=50, groups=200)
        comp.Toggle(0, activateGroup=True)

        with comp.temp_context(groups=300):
            comp.Toggle(0.1, activateGroup=True)

        assert comp.groups == [100, 200]
        assert 200 in comp.triggers[0][P.GROUPS]
        assert 300 in comp.triggers[1][P.GROUPS]
        assert 200 not in comp.triggers[1][P.GROUPS]

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
        """Duplicate groups in a list are detected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError, match="Duplicate"):
            comp._flatten_groups([100, 200, 100])

    def test_flatten_detects_duplicates_across_args(self):
        """Duplicate groups across multiple args are detected"""
        comp = Component("Test", 100)
        with pytest.raises(ValueError, match="Duplicate"):
            comp._flatten_groups(100, [200, 100])


# ============================================================================
# INSTANT PATTERNS - Spawn Order Requirement
# ============================================================================

class TestInstantPatternSpawnOrderRequirement:
    """Test that instant patterns require spawn order"""

    def test_arc_without_spawn_order_rejected(self):
        """Arc without spawn order raises RuntimeError (no PointerCircle)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200).assert_spawn_order(True)
        # No SetPointerCircle called - should fail
        with pytest.raises(RuntimeError) as exc:
            caller.instant.Arc(
                time=0, comp=comp, bullet=lib.bullet1,
                numBullets=5, angle=150
            )
        assert_error(exc, "requires an active PointerCircle")

    def test_radial_without_spawn_order_rejected(self):
        """Radial without spawn order raises RuntimeError (no PointerCircle)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200).assert_spawn_order(True)
        # No SetPointerCircle called - should fail
        with pytest.raises(RuntimeError) as exc:
            caller.instant.Radial(
                time=0, comp=comp, bullet=lib.bullet1,
                spacing=30
            )
        assert_error(exc, "requires an active PointerCircle")

    def test_line_without_spawn_order_rejected(self):
        """Line without spawn order raises ValueError"""
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
    """Test Arc pattern validation - complex math that could break"""

    def test_arc_odd_bullets_fractional_center_accepted(self):
        """Odd bullets with fractional centerAt is now accepted (rounds to 1/3 degree)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Arc(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=5, angle=150, centerAt=45.5
        )

    def test_arc_even_bullets_any_center_accepted(self):
        """Even bullets with any centerAt is now accepted (rounds to 1/3 degree)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Arc(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=4, angle=60, centerAt=0
        )

    def test_arc_fractional_center_accepted(self):
        """Fractional centerAt is now accepted (rounds to 1/3 degree)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Arc(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=4, angle=120, centerAt=45.5
        )

    def test_arc_angle_zero_rejected(self):
        """Angle of 0 is rejected"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.instant.Arc(
                time=0, comp=comp, bullet=lib.bullet1,
                numBullets=5, angle=0
            )
        assert_error(exc, "angle", "0", "360")

    def test_arc_angle_above_360_rejected(self):
        """Angle above 360 is rejected"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.instant.Arc(
                time=0, comp=comp, bullet=lib.bullet1,
                numBullets=1, angle=361
            )
        assert_error(exc, "angle", "360")

    def test_arc_valid_angle_accepted(self):
        """Valid angle is accepted"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - angle of 360 is valid
        caller.instant.Arc(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=10, angle=360
        )

    def test_arc_any_centerAt_accepted(self):
        """Any centerAt value is accepted (rounds to 1/3 degree precision)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Arc(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=5, angle=150, centerAt=45.3, _radialBypass=True
        )


# ============================================================================
# INSTANT RADIAL PATTERN - Validation
# ============================================================================

class TestInstantRadialValidation:
    """Test Radial pattern validation"""

    def test_radial_neither_spacing_nor_numbullets_rejected(self):
        """Must provide either spacing or numBullets"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.instant.Radial(
                time=0, comp=comp, bullet=lib.bullet1
            )
        assert_error(exc, "must provide", "spacing", "numBullets")

    def test_radial_mismatched_spacing_and_numbullets_rejected(self):
        """spacing and numBullets must be consistent"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.instant.Radial(
                time=0, comp=comp, bullet=lib.bullet1,
                numBullets=12, spacing=20  # 360/20 = 18, not 12
            )
        assert_error(exc, "don't match")

    def test_radial_any_spacing_accepted(self):
        """Any spacing value is accepted (rounds to 1/3 degree precision)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Radial(
            time=0, comp=comp, bullet=lib.bullet1,
            spacing=7  # 7 is not a factor of 360, but that's OK now
        )

    def test_radial_any_numbullets_accepted(self):
        """Any numBullets value is accepted (rounds to 1/3 degree precision)"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        # Should not raise - new system rounds to 1/3 degree precision
        caller.instant.Radial(
            time=0, comp=comp, bullet=lib.bullet1,
            numBullets=7  # 7 is not a factor of 360, but that's OK now
        )


# ============================================================================
# INSTANT LINE PATTERN - Validation
# ============================================================================

class TestInstantLineValidation:
    """Test Instant Line pattern validation"""

    def test_line_fastest_zero_rejected(self):
        """fastestTime must be positive"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()
        comp.set_context(target=enums.EMPTY_EMITTER)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        caller = Component("Caller", 200)
        with pytest.raises(ValueError) as exc:
            caller.instant.Line(
                time=0, comp=comp, emitter=50, targetDir=90, bullet=lib.bullet2,
                numBullets=5, fastestTime=0, slowestTime=2.0, dist=100
            )
        assert_error(exc, "positive", "0")

    def test_line_slowest_not_greater_than_fastest_rejected(self):
        """slowestTime must be greater than fastestTime"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()
        comp.set_context(target=enums.EMPTY_EMITTER)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        caller = Component("Caller", 200)
        with pytest.raises(ValueError) as exc:
            caller.instant.Line(
                time=0, comp=comp, emitter=50, targetDir=90, bullet=lib.bullet2,
                numBullets=5, fastestTime=2.0, slowestTime=1.0, dist=100
            )
        assert_error(exc, "greater than", "2.0", "1.0")

    def test_line_too_few_bullets_rejected(self):
        """numBullets must be at least 3"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()
        comp.set_context(target=enums.EMPTY_EMITTER)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        caller = Component("Caller", 200)
        with pytest.raises(ValueError) as exc:
            caller.instant.Line(
                time=0, comp=comp, emitter=50, targetDir=90, bullet=lib.bullet2,
                numBullets=2, fastestTime=0.5, slowestTime=2.0, dist=100
            )
        assert_error(exc, "numBullets must be at least 3")


# ============================================================================
# TIMED PATTERNS - Validation
# ============================================================================

class TestTimedRadialWaveValidation:
    """Test RadialWave pattern validation"""

    def test_radial_wave_zero_waves_rejected(self):
        """waves must be at least 1"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.timed.RadialWave(
                time=0, comp=comp, bullet=lib.bullet1,
                waves=0, numBullets=12
            )
        assert_error(exc, "waves must be at least 1")

    def test_radial_wave_single_wave_rejected(self):
        """Single wave should use instant.Radial instead"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.timed.RadialWave(
                time=0, comp=comp, bullet=lib.bullet1,
                waves=1, numBullets=12
            )
        assert_error(exc, "use instant.Radial")

    def test_radial_wave_negative_interval_rejected(self):
        """interval must be non-negative"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)
        comp.set_context(target=enums.EMPTY_TARGET_GROUP)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        setup_pointer_circle(caller)
        with pytest.raises(ValueError) as exc:
            caller.timed.RadialWave(
                time=0, comp=comp, bullet=lib.bullet1,
                waves=3, interval=-0.5, numBullets=12
            )
        assert_error(exc, "non-negative", "-0.5")


class TestTimedLineValidation:
    """Test Timed Line pattern validation"""

    def test_timed_line_too_few_bullets_rejected(self):
        """numBullets must be at least 2"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        with pytest.raises(ValueError, match="numBullets must be at least 2"):
            caller.timed.Line(
                time=0, comp=comp, targetDir=90, bullet=lib.bullet2,
                numBullets=1, spacing=0.5, t=1.0, dist=100
            )

    def test_timed_line_negative_spacing_rejected(self):
        """spacing must be non-negative"""
        comp = Component("Test", 100).assert_spawn_order(True)
        comp.set_context(target=enums.EMPTY_BULLET)
        comp.Toggle(0, activateGroup=True)

        caller = Component("Caller", 200)
        with pytest.raises(ValueError, match="spacing must be non-negative"):
            caller.timed.Line(
                time=0, comp=comp, targetDir=90, bullet=lib.bullet2,
                numBullets=5, spacing=-0.5, t=1.0, dist=100
            )


# ============================================================================
# METHOD CHAINING - Return Self
# ============================================================================

class TestMethodChainingReturnsSelf:
    """Test that trigger methods return self for chaining"""

    def test_spawn_returns_self(self):
        comp = Component("Test", 100)
        result = comp.Spawn(0, 50, spawnOrdered=False)
        assert result is comp

    def test_toggle_returns_self(self):
        comp = Component("Test", 100)
        comp.set_context(target=50)
        result = comp.Toggle(0, activateGroup=True)
        assert result is comp

    def test_count_returns_self(self):
        comp = Component("Test", 100)
        comp.set_context(target=50)
        result = comp.Count(0, item_id=100, count=5, activateGroup=True)
        assert result is comp

    def test_pickup_returns_self(self):
        comp = Component("Test", 100)
        result = comp.Pickup(0, item_id=5, count=50, override=False)
        assert result is comp

    def test_pickup_modify_returns_self(self):
        comp = Component("Test", 100)
        result = comp.PickupModify(0, item_id=5, factor=1.5, multiply=True)
        assert result is comp

    def test_move_by_returns_self(self):
        comp = Component("Test", 100)
        comp.set_context(target=50)
        result = comp.MoveBy(0, dx=10, dy=20, t=1.0)
        assert result is comp

    def test_rotate_returns_self(self):
        comp = Component("Test", 100)
        comp.set_context(target=50)
        result = comp.Rotate(0, angle=90, t=1.0)
        assert result is comp

    def test_stop_returns_self(self):
        comp = Component("Test", 100)
        result = comp.Stop(0, target=50)
        assert result is comp

    def test_pause_returns_self(self):
        comp = Component("Test", 100)
        result = comp.Pause(0, target=50)
        assert result is comp

    def test_resume_returns_self(self):
        comp = Component("Test", 100)
        result = comp.Resume(0, target=50)
        assert result is comp

    def test_chain_preserves_trigger_count(self):
        """Chained calls accumulate triggers"""
        comp = Component("Test", 100)
        comp.set_context(target=50)
        (comp
            .Toggle(0, activateGroup=True)
            .Alpha(0.1, opacity=50)
            .MoveBy(0.2, dx=10, dy=20)
        )
        assert len(comp.triggers) == 3

    def test_group_context_returns_self(self):
        """Group context methods support chaining"""
        comp = Component("Test", 100)
        result = comp.set_context(groups=200).set_context(target=50).Toggle(0, activateGroup=True).clear_context()
        assert result is comp
        assert len(comp.triggers) == 1


# ============================================================================
# COMPONENT TARGET ACCEPTS COMPONENT OBJECTS
# ============================================================================

class TestGeneralComponentFeatures:

    # Test that methods accept Component objects as targets
    def test_spawn_accepts_component(self):
        """Spawn can target a Component directly"""
        target_comp = Component("Target", 150)
        comp = Component("Test", 100)
        comp.Spawn(0, target_comp, spawnOrdered=False)
        trigger = comp.triggers[0]
        assert trigger[P.TARGET] == 150

    def test_toggle_accepts_component(self):
        """Toggle can target a Component directly"""
        target_comp = Component("Target", 150)
        comp = Component("Test", 100)
        comp.set_context(target=target_comp.caller)
        comp.Toggle(0, activateGroup=True)
        trigger = comp.triggers[0]
        assert trigger[P.TARGET] == target_comp.caller

    def test_component_without_spawn_order_warning(self):
        """Component without requireSpawnOrder gives warning on export"""
        lib.all_components.clear()
        comp = Component("Test", 100)
        comp.set_context(target=50)
        comp.Toggle(0, activateGroup=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lib.save_all(filename="testing")

        assert_warning(w, "spawn order", "Test")


# ============================================================================
# GET_TRIGGERS & HAS_TRIGGER_PROPERTIES
# ============================================================================

class TestGetTriggersMethod:
    """Test Component.get_triggers() and has_trigger_properties()"""

    def test_get_triggers_single_property(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        comp.Pickup(0, item_id=15, count=100, override=False)
        comp.set_context(target=200)
        comp.Count(0, item_id=12, count=10, activateGroup=True)
        comp.clear_context()

        result = comp.get_triggers({P.ITEM_ID: 12})
        assert len(result) == 2
        assert all(t[P.ITEM_ID] == 12 for t in result)

    def test_get_triggers_multiple_properties(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        comp.Pickup(0, item_id=12, count=100, override=False)

        result = comp.get_triggers({P.ITEM_ID: 12, P.PICKUP_COUNT: 50})
        assert len(result) == 1

    def test_get_triggers_wildcard_any(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=11, count=123, override=True)
        comp.PickupModify(0, item_id=11, factor=1.45, multiply=True)

        result = comp.get_triggers({P.PICKUP_MULTIPLY_DIVIDE: Any})
        assert len(result) == 2

    def test_get_triggers_by_object_id(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        comp.set_context(target=300)
        comp.Toggle(0, activateGroup=True)
        comp.clear_context()

        result = comp.get_triggers({P.OBJ_ID: enums.ObjectID.PICKUP})
        assert len(result) == 1

    def test_get_triggers_no_match(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        assert len(comp.get_triggers({P.ITEM_ID: 999})) == 0

    def test_get_triggers_returns_references(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)
        assert comp.get_triggers({P.ITEM_ID: 12})[0] is comp.triggers[0]


class TestHasTriggerPropertiesMethod:

    def test_has_trigger_properties_match(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)

        assert comp.has_trigger_properties({P.ITEM_ID: 12}) is True
        assert comp.has_trigger_properties({P.ITEM_ID: 999}) is False

    def test_has_trigger_properties_multiple(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)

        assert comp.has_trigger_properties({P.ITEM_ID: 12, P.PICKUP_COUNT: 50}) is True
        assert comp.has_trigger_properties({P.ITEM_ID: 12, P.PICKUP_COUNT: 999}) is False

    def test_has_trigger_properties_wildcard(self):
        comp = Component("Test", 100, 5)
        comp.PickupModify(0, item_id=11, factor=1.45, multiply=True)

        assert comp.has_trigger_properties({P.PICKUP_MULTIPLY_DIVIDE: Any}) is True

    def test_has_trigger_properties_empty_dict_rejected(self):
        comp = Component("Test", 100, 5)
        with pytest.raises(ValueError) as exc_info:
            comp.has_trigger_properties({})
        assert_error(exc_info, "empty")

    def test_consistency_with_get_triggers(self):
        comp = Component("Test", 100, 5)
        comp.Pickup(0, item_id=12, count=50, override=True)

        query = {P.ITEM_ID: 12}
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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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
        lib.all_components.clear()

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

        Why: Multitarget creates binary spawn trees, which would multiply
        spawn triggers exponentially, causing spawn limit violations.
        """
        lib.all_components.clear()

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
    """Tests that closely timed triggers are grouped or separated correctly"""

    def test_triggers_beyond_tolerance_are_separated(self):
        """
        Triggers beyond tolerance should be in separate groups (no violation).

        Setup: B has 3 spawns at time 0.0s, ~0.004s, ~0.008s
        Expected: Only first two are grouped (~0.004s converts to ~1.25 studs, within ~1.3 tolerance)z ;
            Third is separate (~0.008s converts to ~2.49 studs, beyond tolerance)
            Only 2 simultaneous spawns → violation
        """
        lib.all_components.clear()

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
        lib.all_components.clear()

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
    """Test solid group enforcement to prevent position groups from being used as spawnable groups"""

    def setup_method(self):
        """Clear solid groups and components before each test"""
        lib.solid_groups_to_enforce.clear()
        lib.all_components.clear()

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

    def test_different_group_as_component_caller_allowed(self):
        """A different group as component caller is fine"""
        comp = Component("GoodComponent", callerGroup=100)
        lib.all_components.append(comp)
        
        # Should not raise - 200 is not the caller
        lib._validate_solid_groups(200)

    def test_solid_group_as_move_target_allowed(self):
        """A solid group as Move target (via set_context) is allowed"""
        comp = Component("GoodComponent", callerGroup=50)
        comp.set_context(target=100)
        comp.MoveBy(0, dx=10, dy=10, t=1)
        lib.all_components.append(comp)
        
        # Should not raise - 100 is the target, not in groups array
        lib._validate_solid_groups(100)

    def test_keyframe_obj_trigger_exempt(self):
        """KeyframeObj triggers are exempt from solid group validation"""
        comp = Component("ScaleComponent", callerGroup=50)
        comp.set_context(target=100)
        comp.Scale(0, factor=2.0, t=1.0, hold=0.5)
        lib.all_components.append(comp)
        
        # Should not raise even though KEYFRAME_OBJ might reference group 100
        lib._validate_solid_groups(100)

    def test_pointer_obj_trigger_exempt(self):
        """PointerObj triggers are exempt from solid group validation"""
        comp = Component("PointerComponent", callerGroup=50)
        comp.set_context(target=100)
        trigger = comp.create_trigger(enums.ObjectID.POINTER_OBJ, x=0, target=100)
        trigger[enums.Properties.GROUPS] = [100]
        lib.all_components.append(comp)
        
        # Should not raise because POINTER_OBJ is in solid_obj_ids exemption set
        lib._validate_solid_groups(100)

    def test_move_towards_enforces_target_as_solid(self):
        """MoveTowards calls enforce_solid_groups on its target"""
        lib.solid_groups_to_enforce.clear()
        comp = Component("Bullet", callerGroup=500)
        comp.set_context(target=600)
        comp.MoveTowards(0, targetDir=90, t=1.0, dist=100)
        
        assert 600 in lib.solid_groups_to_enforce

    def test_goto_group_enforces_target_as_solid(self):
        """GotoGroup calls enforce_solid_groups on its target"""
        lib.solid_groups_to_enforce.clear()
        comp = Component("Bullet", callerGroup=500)
        comp.set_context(target=600)
        comp.GotoGroup(0, location=700, t=1.0)
        
        assert 600 in lib.solid_groups_to_enforce

    def test_point_to_group_enforces_targetdir_as_solid(self):
        """PointToGroup calls enforce_solid_groups on targetDir"""
        lib.solid_groups_to_enforce.clear()
        comp = Component("Bullet", callerGroup=500)
        comp.set_context(target=600)
        comp.PointToGroup(0, targetDir=700, t=1.0)
        
        assert 700 in lib.solid_groups_to_enforce

    def test_scale_enforces_target_as_solid(self):
        """Scale calls enforce_solid_groups on its target"""
        lib.solid_groups_to_enforce.clear()
        comp = Component("Bullet", callerGroup=500)
        comp.set_context(target=600)
        comp.Scale(0, factor=2.0, t=1.0, hold=0.5)
        
        assert 600 in lib.solid_groups_to_enforce

    def test_complex_scene_conflict_detected(self):
        """Test validation with multiple components - conflict should be detected"""
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
        """Test that properly separated solid and trigger groups validate successfully"""
        bullet = Component("Bullet", callerGroup=500)
        bullet.set_context(target=600)
        bullet.MoveTowards(0, targetDir=1000, t=1.0, dist=100)
        lib.all_components.append(bullet)
        
        pattern = Component("Pattern", callerGroup=501)
        pattern.Spawn(0, target=2000, spawnOrdered=False)
        lib.all_components.append(pattern)
        
        # Should not raise - groups are properly separated
        lib._validate_solid_groups()

    def test_multiple_components_same_solid_group_allowed(self):
        """Multiple components can use the same solid group as target"""
        bullet1 = Component("Bullet1", callerGroup=500)
        bullet1.set_context(target=600)
        bullet1.MoveTowards(0, targetDir=1000, t=1.0, dist=100)
        lib.all_components.append(bullet1)
        
        bullet2 = Component("Bullet2", callerGroup=501)
        bullet2.set_context(target=601)
        bullet2.MoveTowards(0, targetDir=1000, t=1.0, dist=100)
        lib.all_components.append(bullet2)
        
        # Both use 1000 as targetDir - should be fine
        lib._validate_solid_groups()

    def test_empty_enforcement_set_passes(self):
        """Validating with empty enforcement set should pass"""
        comp = Component("Component", callerGroup=100)
        comp.Spawn(0, target=200, spawnOrdered=False)
        lib.all_components.append(comp)
        
        lib._validate_solid_groups()

    def test_specific_groups_parameter(self):
        """Test _validate_solid_groups with specific groups parameter"""
        comp = Component("Component", callerGroup=100)
        comp.set_context(target=200)
        comp.Toggle(0, activateGroup=True)
        lib.all_components.append(comp)
        
        lib.solid_groups_to_enforce.add(100)
        
        # Calling with specific group 200 should pass (100 is bad but not checked)
        lib._validate_solid_groups(200)
        
        # Calling with specific group 100 should fail
        with pytest.raises(ValueError):
            lib._validate_solid_groups(100)
