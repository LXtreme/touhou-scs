from touhou_scs import enums as e
from touhou_scs import lib
from touhou_scs.component import Component
from touhou_scs.lib import Stage, enemy1, save_all
from touhou_scs.misc import add_disable_all_bullets, add_enemy_collisions, add_plr_collisions
from touhou_scs.utils import unknown_g

if __name__ != "__main__":
    print("Don't import this! exiting.")
    exit()

main = (Component("Main", 36, 7)
    .assert_spawn_order(False)
    .Spawn(0, lib.Stage.stage1.caller, True)
)

c1 = lib.circle1

# ===========================================================================
# POSITIONING POINTERS
# These are invisible groups we move around to use as coordinates
# ===========================================================================

top_left = lib.pointer.next()
top_right = lib.pointer.next()
middle_test = lib.pointer.next()

# Setup positioning pointers
pos_setup = (Component("Position Setup", unknown_g(), 11)
    .assert_spawn_order(True)
    .set_context(target=top_left)
        .SetPosition(0, x=0,   y=420)
    .set_context(target=top_right)
        .SetPosition(0, x=360, y=420)
    .set_context(target=middle_test)
        .SetPosition(0, x=180, y=300)
    .clear_context()
)


# ===========================================================================
# BULLET COMPONENTS
# ===========================================================================

# Test bullet for 1080 precision test
test_bullet = (Component("TestBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.3, opacity=100)
        .Scale(0, factor=2, t=0.3, reverse=True)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(0.2, e.EMPTY_TARGET_GROUP, t=6, dist=450, type=e.Easing.EASE_IN, rate=1.6)
    .clear_context()
)

# ===========================================================================
# TEST PATTERNS
# =========================================================================

test_enemy_g = enemy1.next()
test_enemy = (Component("TestEnemy", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=test_enemy_g)
        .GotoGroup(0, middle_test)
    .clear_context()
)
(test_enemy
    .pointer.SetPointerCircle(0.5, location=test_enemy_g, follow=True)
    # .set_context(target=test_enemy.pointer.pc.all)
    #     .MoveBy(0.6, dx=-200, dy=0, t=15)
    # .clear_context()
    # Test 1: Basic radial with 5 bullets at 0°
    .instant.Radial(1.0, test_bullet, lib.bullet1, numBullets=5, centerAt=0)
    # # Test 2: Same radial pattern (tests pointer reuse)
    .instant.Radial(1.5, test_bullet, lib.bullet1, numBullets=5, centerAt=0)
    # # Test 3: Different centerAt (90°)
    .instant.Radial(2.0, test_bullet, lib.bullet1, numBullets=5, centerAt=90)
    # # Test 4: Different numBullets (8 bullets)
    .instant.Radial(2.5, test_bullet, lib.bullet1, numBullets=8, centerAt=0)
    # # Test 5: High bullet count (16 bullets at 180°)
    .instant.Radial(3.0, test_bullet, lib.bullet1, numBullets=16, centerAt=180)
    # # Test 6: Low bullet count (3 bullets at 270°)
    .instant.Radial(3.5, test_bullet, lib.bullet1, numBullets=3, centerAt=270)
    # # Test 7: Arc pattern - 90° arc with 7 bullets at 45°
    .instant.Arc(4.0, test_bullet, lib.bullet1, angle=90, numBullets=7, centerAt=45)
    # # Test 8: Arc pattern - 180° arc with 10 bullets at 180°
    .instant.Arc(4.5, test_bullet, lib.bullet1, angle=180, numBullets=10, centerAt=180)
    # # Test 9: Narrow arc - 30° with 5 bullets at 0°
    .instant.Arc(5.0, test_bullet, lib.bullet1, angle=30, numBullets=5, centerAt=0)
    # # Test 10: Wide arc - 270° with 15 bullets at 270°
    .instant.Arc(5.5, test_bullet, lib.bullet1, angle=270, numBullets=15, centerAt=270)
    # # Test 11: Two bullets at odd angle (33.333°)
    .instant.Radial(6.0, test_bullet, lib.bullet1, numBullets=2, centerAt=33.333)
    # # Test 12: Prime number bullets (13) at odd angle (127.5°)
    .instant.Radial(6.5, test_bullet, lib.bullet1, numBullets=13, centerAt=127.5)
    # # Test 13: Large radial (24 bullets at 0°) - tests many pointer allocations
    .instant.Radial(7.0, test_bullet, lib.bullet1, numBullets=24, centerAt=0)
    # # Test 14: Full arc (360° with 12 bullets at 0°) - should behave like radial
    .instant.Arc(7.5, test_bullet, lib.bullet1, angle=360, numBullets=12, centerAt=0)
    .pointer.CleanPointerCircle()
)

# ===========================================================================
# SPAWN ENEMIES ON STAGE
# ===========================================================================

enemy1.spawn_enemy(Stage.stage1, 1.0, test_enemy, 51, test_enemy_g)

# ===========================================================================
# STAGE SETUP
# ===========================================================================

Stage.stage1.Spawn(0, pos_setup.caller, True)

# ===========================================================================
# SYSTEM SETUP
# ===========================================================================

add_enemy_collisions()
add_disable_all_bullets()
add_plr_collisions()
save_all()
