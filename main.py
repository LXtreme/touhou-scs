from touhou_scs import enums as e
from touhou_scs import lib
from touhou_scs.component import BulletAlloc, Component
from touhou_scs.lib import Stage, enemy1, rgb, save_all
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
# ===========================================================================

top_left = lib.pointer.next()
top_right = lib.pointer.next()
middle_test = lib.pointer.next()
middle_left = lib.pointer.next()
middle_right = lib.pointer.next()
side_left = lib.pointer.next()
side_right = lib.pointer.next()

pos_setup = (Component("Position Setup", unknown_g(), 11)
    .assert_spawn_order(True)
    .set_context(target=top_left)
        .SetPosition(0, x=0,   y=420)
    .set_context(target=top_right)
        .SetPosition(0, x=360, y=420)
    .set_context(target=middle_test)
        .SetPosition(0, x=180, y=300)
    .set_context(target=middle_left)
        .SetPosition(0, x=90, y=280)
    .set_context(target=middle_right)
        .SetPosition(0, x=180+90, y=280)
    .set_context(target=side_left)
        .SetPosition(0, x=60, y=260)
    .set_context(target=side_right)
        .SetPosition(0, x=360-60, y=260)
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
        .Pulse(0, rgb(255,0,0), t=4.5)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.3, opacity=100)
        .Scale(0, factor=2, t=0.4, reverse=True)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(e.TICK, e.EMPTY_TARGET_GROUP, t=4.5, dist=450, type=e.Easing.EASE_IN, rate=1.4)
        .Pulse(0.2, rgb(255,105,5), fadeIn=0.1, t=0, fadeOut=0.2)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(-e.TICK, False)
        .Toggle(1, True)
    .clear_context()
)

test_bullet3 = (Component("TestBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Pulse(0, rgb(105,0,205), t=5)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.6, opacity=100)
        .Scale(0, factor=2, t=0.4, reverse=True)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(e.TICK, e.EMPTY_TARGET_GROUP, t=4, dist=450, type=e.Easing.EASE_IN, rate=1.6)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(-e.TICK, False)
        .Toggle(1, True)
    .clear_context()
)


test_bullet4 = (Component("TestBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Pulse(0, rgb(0,50,255), t=5)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.6, opacity=100)
        .Scale(0, factor=2, t=0.4, reverse=True)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(e.TICK, e.EMPTY_TARGET_GROUP, t=3, dist=450, type=e.Easing.EASE_IN, rate=1.6)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(-e.TICK, False)
        .Toggle(1, True)
    .clear_context()
)


test_bullet2 = (Component("TestBullet", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=e.EMPTY_BULLET)
        .GotoGroup(0, e.EMPTY_EMITTER)
        .Toggle(e.TICK, True)
        .Pulse(0, rgb(50, 70, 230), t=4.1)
        .Alpha(0, t=0, opacity=0)
        .Alpha(e.TICK, t=0.3, opacity=100)
        .Scale(0, factor=1.7, t=0.4, hold=2)
        .PointToGroup(e.TICK, e.EMPTY_TARGET_GROUP)
        .MoveTowards(e.TICK, e.EMPTY_TARGET_GROUP, t=2, dist=450, type=e.Easing.EASE_IN, rate=1.6)
        .Pulse(2, rgb(205,90,250), fadeIn=0.1, t=0, fadeOut=0.4)
    .set_context(target=e.EMPTY_COLLISION)
        .Toggle(-e.TICK, False)
        .Toggle(1, True)
    .clear_context()
)

# ===========================================================================
# TEST PATTERNS
# ===========================================================================

BulletAlloc.start()

test_enemy_g = enemy1.next()
test_enemy = (Component("TestEnemy", unknown_g(), 5)
    .assert_spawn_order(True)
    .set_context(target=test_enemy_g)
        .GotoGroup(0, middle_test)
    .clear_context()
    .pointer.SetPointerCircle(0.4, location=test_enemy_g, follow=True)
)
(test_enemy
    .set_context(target=test_enemy.pointer.pc.all)
        .MoveBy(0.5, dx=40, dy=0, t=0)
        .Rotate(0.55, angle=3060, t=20)
        .Rotate(0.55, center=test_enemy_g, angle=3060, t=10, type=1, rate=1.5)
        .Rotate(0.6, center=test_enemy_g, angle=-5600, t=10, type=2, rate=2.5)
        .Rotate(1.6, center=test_enemy_g, angle=-5600, t=4, type=1, rate=2.5)
        .Rotate(8.5, center=test_enemy_g, angle=1560, t=8, type=2, rate=2.5)
        .Rotate(6.5, center=test_enemy_g, angle=3060, t=10, type=1, rate=2.5)
        .Rotate(6.5, center=test_enemy_g, angle=-760, t=14, type=2, rate=2.5)
        .Rotate(14.5, center=test_enemy_g, angle=3060, t=10, type=2, rate=2.5)
        .Rotate(15.5, center=test_enemy_g, angle=-7600, t=14, type=1, rate=2.5)
        .Rotate(16.5, center=test_enemy_g, angle=1560, t=4, type=2, rate=2.5)
    .clear_context()
    .timed.RadialWave(1.0, test_bullet, lib.bullet3, waves=48, interval=0.5, numBullets=30)
    .pointer.CleanPointerCircle()
    
    
    
    .pointer.SetPointerCircle(5, location=middle_left, follow=True)
    .set_context(target=test_enemy.pointer.pc.all)
        .MoveBy(5+0.5, dx=-60, dy=0, t=0)
        .Rotate(5+0.55, angle=3060, t=20)
        .Rotate(5+0.55, center=middle_left, angle=3060, t=10, type=1, rate=1.5)
        .Rotate(5+0.6, center=middle_left, angle=-5600, t=10, type=2, rate=2.5)
        .Rotate(5+1.6, center=middle_left, angle=-5600, t=4, type=1, rate=2.5)
        .Rotate(5+8.5, center=middle_left, angle=1560, t=8, type=2, rate=2.5)
        .Rotate(5+6.5, center=middle_left, angle=3060, t=10, type=1, rate=2.5)
        .Rotate(5+6.5, center=middle_left, angle=-760, t=14, type=2, rate=2.5)
        .Rotate(5+14.5, center=middle_left, angle=3060, t=10, type=2, rate=2.5)
        .Rotate(5+15.5, center=middle_left, angle=-7600, t=14, type=1, rate=2.5)
        .Rotate(5+16.5, center=middle_left, angle=1560, t=4, type=2, rate=2.5)
    .clear_context()
    .timed.RadialWave(5.5, test_bullet3, lib.bullet3, waves=20, interval=1, numBullets=14)
    .pointer.CleanPointerCircle()
    
    .pointer.SetPointerCircle(4.5, location=middle_right, follow=True)
    .set_context(target=test_enemy.pointer.pc.all)
        .MoveBy(4.5+0.5, dx=-60, dy=0, t=0)
        .Rotate(4.5+0.55, angle=3060, t=20)
        .Rotate(4.5+0.55, center=middle_right, angle=3060, t=10, type=1, rate=1.5)
        .Rotate(4.5+0.6, center=middle_right, angle=-5600, t=10, type=2, rate=2.5)
        .Rotate(4.5+1.6, center=middle_right, angle=-5600, t=4, type=1, rate=2.5)
        .Rotate(4.5+8.5, center=middle_right, angle=1560, t=8, type=2, rate=2.5)
        .Rotate(4.5+6.5, center=middle_right, angle=3060, t=10, type=1, rate=2.5)
        .Rotate(4.5+6.5, center=middle_right, angle=-760, t=14, type=2, rate=2.5)
        .Rotate(4.5+14.5, center=middle_right, angle=3060, t=10, type=2, rate=2.5)
        .Rotate(4.5+15.5, center=middle_right, angle=-7600, t=14, type=1, rate=2.5)
        .Rotate(4.5+16.5, center=middle_right, angle=1560, t=4, type=2, rate=2.5)
    .clear_context()
    .timed.RadialWave(5, test_bullet3, lib.bullet3, waves=20, interval=1, numBullets=14)
    .pointer.CleanPointerCircle()
    
    
    
    .pointer.SetPointerCircle(4, location=side_left, follow=True)
    .set_context(target=test_enemy.pointer.pc.all)
        .MoveBy(9+0.5, dx=-20, dy=0, t=0)
        .Rotate(9+0.55, angle=3060, t=20)
        .Rotate(9+0.55, center=side_left, angle=3060, t=10, type=1, rate=1.5)
        .Rotate(9+0.6, center=side_left, angle=-5600, t=10, type=2, rate=2.5)
        .Rotate(9+1.6, center=side_left, angle=-5600, t=4, type=1, rate=2.5)
        .Rotate(9+8.5, center=side_left, angle=1560, t=8, type=2, rate=2.5)
        .Rotate(9+6.5, center=side_left, angle=3060, t=10, type=1, rate=2.5)
        .Rotate(9+6.5, center=side_left, angle=-760, t=14, type=2, rate=2.5)
        .Rotate(9+14.5, center=side_left, angle=3060, t=10, type=2, rate=2.5)
        .Rotate(9+15.5, center=side_left, angle=-7600, t=14, type=1, rate=2.5)
        .Rotate(9+16.5, center=side_left, angle=1560, t=4, type=2, rate=2.5)
    .clear_context()
    .timed.RadialWave(9.5, test_bullet4, lib.bullet3, waves=10, interval=1.8, numBullets=40)
    .pointer.CleanPointerCircle()
    
    .pointer.SetPointerCircle(3.5, location=side_right, follow=True)
    .set_context(target=test_enemy.pointer.pc.all)
        .MoveBy(9.2+0.5, dx=-20, dy=0, t=0)
        .Rotate(9.2+0.55, angle=3060, t=20)
        .Rotate(9.2+0.55, center=side_right, angle=3060, t=10, type=1, rate=1.5)
        .Rotate(9.2+0.6, center=side_right, angle=-5600, t=10, type=2, rate=2.5)
        .Rotate(9.2+1.6, center=side_right, angle=-5600, t=4, type=1, rate=2.5)
        .Rotate(9.2+8.5, center=side_right, angle=1560, t=8, type=2, rate=2.5)
        .Rotate(9.2+6.5, center=side_right, angle=3060, t=10, type=1, rate=2.5)
        .Rotate(9.2+6.5, center=side_right, angle=-760, t=14, type=2, rate=2.5)
        .Rotate(9.2+14.5, center=side_right, angle=3060, t=10, type=2, rate=2.5)
        .Rotate(9.2+15.5, center=side_right, angle=-7600, t=14, type=1, rate=2.5)
        .Rotate(9.2+16.5, center=side_right, angle=1560, t=4, type=2, rate=2.5)
    .clear_context()
    .timed.RadialWave(9.7, test_bullet4, lib.bullet3, waves=10, interval=1.8, numBullets=40)
    .pointer.CleanPointerCircle()
    
    
    .pointer.SetPointerCircle(3, location=test_enemy_g, follow=True)
    .set_context(target=test_enemy.pointer.pc.all)
        .Rotate(4, center=test_enemy_g, angle=11.25, t=0)
    .clear_context()
    .timed.RadialWave(15, test_bullet2, lib.bullet2, waves=30, interval=0.25, numBullets=16)
    .pointer.CleanPointerCircle()
)

BulletAlloc.resolve()

enemy1.spawn_enemy(Stage.stage1, 1.0, test_enemy, 150, test_enemy_g)

Stage.stage1.Spawn(0, pos_setup.caller, True)

add_enemy_collisions()
add_disable_all_bullets()
add_plr_collisions()
save_all()
