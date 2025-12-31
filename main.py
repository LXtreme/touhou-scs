from touhou_scs import enums as e
from touhou_scs import lib
from touhou_scs.component import Component
from touhou_scs.lib import Stage, enemy1, rgb, save_all, HSB
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

# Entry point - left side offscreen
entry_left = lib.pointer.next()[0]
# Entry point - right side offscreen
entry_right = lib.pointer.next()[0]
# Exit point - top left offscreen
exit_top_left = lib.pointer.next()[0]
# Exit point - top right offscreen
exit_top_right = lib.pointer.next()[0]
# Enemy stop position - upper left area
stop_pos_left = lib.pointer.next()[0]
# Enemy stop position - upper right area
stop_pos_right = lib.pointer.next()[0]
# Enemy stop position - upper center
stop_pos_center = lib.pointer.next()[0]
# Miniboss center position
miniboss_pos = lib.pointer.next()[0]

# Setup positioning pointers
pos_setup = (Component("Position Setup", unknown_g(), 11)
    .assert_spawn_order(True)
    # Left entry (offscreen left, mid height)
    .set_context(target=entry_left)
        .SetPosition(0, x=-70, y=310)
    # Right entry (offscreen right, mid height)
    .set_context(target=entry_right)
        .SetPosition(0, x=430, y=310)
    .set_context(target=exit_top_left)
        .SetPosition(0, x=-70, y=460)
    .set_context(target=exit_top_right)
        .SetPosition(0, x=430, y=460)
    .set_context(target=stop_pos_left)
        .SetPosition(0, x=80, y=330)
    .set_context(target=stop_pos_right)
        .SetPosition(0, x=280, y=330)
    .set_context(target=stop_pos_center)
        .SetPosition(0, x=180, y=340)
    # Miniboss position (center, upper area)
    .set_context(target=miniboss_pos)
        .SetPosition(0, x=180, y=310)
    .clear_context()
)

# Spawn position setup at stage start
Stage.stage1.Spawn(0, pos_setup.caller, False)

# ===========================================================================
# BULLET COMPONENTS
# ===========================================================================

# Simple aimed bullet - fires toward player
simple_aimed = (Component("SimpleAimed", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.3, opacity=100)
        .Scale(0, factor=2, t=0.3, reverse=True)
        .PointToGroup(e.TICK, e.PLR)
        .MoveTowards(0.2, e.PLR, t=4, dist=500, type=e.Easing.EASE_IN, rate=1.5)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(0, False)
        .Toggle(0.5, True)
    .clear_context()
)

# Radial bullet - spreads outward from emitter
radial_bullet = (Component("RadialBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.5, opacity=100)
        .Scale(0, factor=2.5, t=0.4, reverse=True)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(0.3, e.EMPTY_TARGET_GROUP, t=3, dist=400, type=e.Easing.EASE_OUT, rate=1.2)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(0, False)
        .Toggle(0.5, True)
    .clear_context()
)

# Fast bullet for miniboss
fast_bullet = (Component("FastBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.2, opacity=100)
        .Scale(0, factor=1.5, t=0.2, reverse=True)
        .Pulse(0, rgb(255, 50, 50), t=10)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(0.15, e.EMPTY_TARGET_GROUP, t=2, dist=500, type=e.Easing.EASE_IN, rate=2)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(0, False)
        .Toggle(0.5, True)
    .clear_context()
)

# Slow homing bullet for miniboss - starts toward angle, then homes to player
homing_bullet = (Component("HomingBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.8, opacity=100)
        .Scale(0, factor=3, t=0.6, reverse=True)
        .Pulse(0, rgb(100, 0, 255), t=10)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(0.5, e.EMPTY_TARGET_GROUP, t=1.5, dist=80, type=e.Easing.EASE_IN_OUT, rate=2)
        .PointToGroup(2, e.PLR, t=0.5)
        .MoveTowards(2.5, e.PLR, t=5, dist=600, type=e.Easing.EASE_IN, rate=1.5)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(0, False)
        .Toggle(0.5, True)
    .clear_context()
)

# Spiral bullet for miniboss finale
spiral_bullet = (Component("SpiralBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.4, opacity=100)
        .Scale(0, factor=2, t=0.5, reverse=True)
        .Pulse(0, HSB(180, 200, 100), t=8)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(0.2, e.EMPTY_TARGET_GROUP, t=4, dist=450, type=e.Easing.NONE, rate=1)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(0, False)
        .Toggle(0.5, True)
    .clear_context()
)

# ===========================================================================
# REGULAR ENEMY ATTACKS (5 enemies)
# ===========================================================================

# Enemy 1 - Enters from left, shoots simple aimed burst, exits top-left
enemy1g = enemy1.next()
enemy1_attack = (Component("Enemy1_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=enemy1g)
        .GotoGroup(0, entry_left)
        .GotoGroup(0.1, stop_pos_left, t=1, type=e.Easing.EASE_OUT, rate=1.5)
    .clear_context())
(enemy1_attack
    .pointer.SetPointerCircle(1.2, c1, location=enemy1g)
    .instant.Radial(1.5, radial_bullet, lib.bullet1, numBullets=12, centerAt=0)
    .instant.Radial(2.0, radial_bullet, lib.bullet1, numBullets=12, centerAt=15)
    .pointer.CleanPointerCircle()
    .set_context(target=enemy1g)
        .GotoGroup(2.8, exit_top_left, t=1.2, type=e.Easing.EASE_IN, rate=1.5)
    .clear_context()
)

# Enemy 2 - Enters from right, shoots aimed lines, exits top-right
enemy2g = enemy1.next()
enemy2_attack = (Component("Enemy2_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=enemy2g)
        .GotoGroup(0, entry_right)
        .GotoGroup(0.1, stop_pos_right, t=1, type=e.Easing.EASE_OUT, rate=1.5)
    .clear_context()
    .instant.Line(1.3, simple_aimed, enemy2g, e.PLR, lib.bullet2, numBullets=8, fastestTime=2, slowestTime=3.5, dist=500)
    .instant.Line(1.8, simple_aimed, enemy2g, e.PLR, lib.bullet2, numBullets=8, fastestTime=2, slowestTime=3.5, dist=500)
    .set_context(target=enemy2g)
        .GotoGroup(2.8, exit_top_right, t=1.2, type=e.Easing.EASE_IN, rate=1.5)
    .clear_context()
)

# Enemy 3 - Enters from left, shoots radial pattern, exits top-left
enemy3g = enemy1.next()
enemy3_attack = (Component("Enemy3_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=enemy3g)
        .GotoGroup(0, entry_left)
        .GotoGroup(0.1, stop_pos_center, t=1.2, type=e.Easing.EASE_OUT, rate=1.3)
    .clear_context())
(enemy3_attack
    .pointer.SetPointerCircle(1.3, c1, location=enemy3g)
    .instant.Radial(1.5, radial_bullet, lib.bullet3, numBullets=18, centerAt=0)
    .instant.Radial(1.9, radial_bullet, lib.bullet3, numBullets=18, centerAt=10)
    .instant.Radial(2.3, radial_bullet, lib.bullet3, numBullets=18, centerAt=20)
    .pointer.CleanPointerCircle()
    .set_context(target=enemy3g)
        .GotoGroup(3.2, exit_top_left, t=1, type=e.Easing.EASE_IN, rate=1.5)
    .clear_context()
)

# Enemy 4 - Enters from right, shoots aimed burst, exits top-right
enemy4g = enemy1.next()
enemy4_attack = (Component("Enemy4_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=enemy4g)
        .GotoGroup(0, entry_right)
        .GotoGroup(0.1, stop_pos_right, t=1, type=e.Easing.EASE_OUT, rate=1.5)
    .clear_context()
    .instant.Line(1.3, simple_aimed, enemy4g, e.PLR, lib.bullet4, numBullets=10, fastestTime=1.5, slowestTime=5, dist=480)
    .instant.Line(2.0, simple_aimed, enemy4g, e.PLR, lib.bullet4, numBullets=10, fastestTime=1.5, slowestTime=5, dist=480)
    .set_context(target=enemy4g)
        .GotoGroup(3.0, exit_top_right, t=1.2, type=e.Easing.EASE_IN, rate=1.5)
    .clear_context()
)

# Enemy 5 - Enters from left, shoots mixed pattern, exits top-left
enemy5g = enemy1.next()
enemy5_attack = (Component("Enemy5_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=enemy5g)
        .GotoGroup(0, entry_left)
        .GotoGroup(0.1, stop_pos_left, t=1, type=e.Easing.EASE_OUT, rate=1.5)
    .clear_context())
(enemy5_attack
    .pointer.SetPointerCircle(1.2, c1, location=enemy5g)
    .instant.Radial(1.4, radial_bullet, lib.bullet1, numBullets=8, centerAt=0)
    .pointer.CleanPointerCircle()
    .instant.Line(1.8, simple_aimed, enemy5g, e.PLR, lib.bullet2, numBullets=6, fastestTime=2, slowestTime=3, dist=450)
    .set_context(target=enemy5g)
        .GotoGroup(2.8, exit_top_left, t=1.2, type=e.Easing.EASE_IN, rate=1.5)
    .clear_context()
)

# ===========================================================================
# MINIBOSS ATTACK
# ===========================================================================

minibossg = enemy1.next()
miniboss_attack = (Component("Miniboss_Attack", unknown_g(), 5)
    .assert_spawn_order(True)
    # Dramatic entrance from top
    .set_context(target=minibossg)
        .GotoGroup(0, e.GAME_CENTER)
        .MoveBy(0, dx=0, dy=280)
        .GotoGroup(0.1, miniboss_pos, t=2, type=e.Easing.EASE_OUT, rate=1.2)
        .Scale(0, factor=1.5, t=2, reverse=True)
    .clear_context())

# Phase 1: Radial waves
(miniboss_attack
    .pointer.SetPointerCircle(2.5, c1, location=minibossg, duration=8)
    .set_context(target=miniboss_attack.pointer.center)
        .MoveBy(3, dx=-60, dy=0, t=2, type=e.Easing.EASE_IN_OUT)
        .MoveBy(5, dx=120, dy=0, t=2, type=e.Easing.EASE_IN_OUT)
        .MoveBy(7, dx=-60, dy=0, t=2, type=e.Easing.EASE_IN_OUT)
    .clear_context()
    .timed.RadialWave(3, fast_bullet, lib.bullet1, numBullets=12, waves=6, interval=0.4, centerAt=0)
    .timed.RadialWave(3.2, fast_bullet, lib.bullet1, numBullets=12, waves=6, interval=0.4, centerAt=15)
    .pointer.CleanPointerCircle()
)

# Phase 2: Homing bullets + aimed lines
(miniboss_attack
    .pointer.SetPointerCircle(7, c1, location=minibossg)
    .instant.Radial(7.5, homing_bullet, lib.bullet3, numBullets=6, centerAt=0)
    .instant.Radial(8.5, homing_bullet, lib.bullet3, numBullets=6, centerAt=30)
    .pointer.CleanPointerCircle()
    .instant.Line(9, simple_aimed, minibossg, e.PLR, lib.bullet2, numBullets=12, fastestTime=1, slowestTime=2.5, dist=500)
    .instant.Line(9.8, simple_aimed, minibossg, e.PLR, lib.bullet2, numBullets=12, fastestTime=1, slowestTime=2.5, dist=500)
)

# Phase 3: Spiral finale with movement
(miniboss_attack
    .pointer.SetPointerCircle(11, c1, location=minibossg, duration=6)
    .set_context(target=miniboss_attack.pointer.center)
        .MoveBy(11.5, dx=-80, dy=-30, t=3, type=e.Easing.EASE_IN_OUT)
        .MoveBy(14.5, dx=80, dy=30, t=3, type=e.Easing.EASE_IN_OUT)
    .clear_context()
    .timed.RadialWave(11.5, spiral_bullet, lib.bullet4, numBullets=24, waves=12, interval=0.3, centerAt=0)
    .pointer.CleanPointerCircle()
)

# Exit after attack
(miniboss_attack
    .set_context(target=minibossg)
        .GotoGroup(18, exit_top_right, t=1.5, type=e.Easing.EASE_IN, rate=2)
    .clear_context()
)

# ===========================================================================
# SPAWN ENEMIES ON STAGE
# ===========================================================================

# Regular enemies spawn every ~2 seconds
enemy1.spawn_enemy(Stage.stage1, 0.5, enemy1_attack, 21, enemy1g)
enemy1.spawn_enemy(Stage.stage1, 2.5, enemy2_attack, 22, enemy2g)
enemy1.spawn_enemy(Stage.stage1, 4.5, enemy3_attack, 23, enemy3g)
enemy1.spawn_enemy(Stage.stage1, 6.5, enemy4_attack, 28, enemy4g)
enemy1.spawn_enemy(Stage.stage1, 8.5, enemy5_attack, 29, enemy5g)

# Miniboss appears after regular enemies
enemy1.spawn_enemy(Stage.stage1, 12, miniboss_attack, 50, minibossg)

# ===========================================================================
# SYSTEM SETUP
# ===========================================================================

add_enemy_collisions()
add_disable_all_bullets()
add_plr_collisions()
save_all()
