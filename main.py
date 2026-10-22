from gmdbuilder import Level
from touhou_scs.utils import init_level, unknown_g

if __name__ != "__main__":
    print("Don't import this! exiting.")
    exit()

level = Level.from_file("touhou scs.gdlvl")
# level.export_to_live_editor()
init_level(level)

from touhou_scs import enums as e
from touhou_scs import lib
from touhou_scs.component import BulletAlloc, Component
from touhou_scs.lib import Stage, yinyang, rgb, save_all
from touhou_scs.misc import add_disable_all_prefabs, add_enemy_collisions, add_plr_collisions, add_pickup_collisions

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

chain_entry  = lib.pointer.next()  # (-60, 420) offscreen top-left
chain_mid_l  = lib.pointer.next()  # (90,  300) enter shooting zone
chain_mid_r  = lib.pointer.next()  # (270, 300) leave shooting zone
chain_exit   = lib.pointer.next()  # (420, 420) offscreen top-right

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
    .set_context(target=chain_entry)
        .SetPosition(0, x=-60, y=420)
    .set_context(target=chain_mid_l)
        .SetPosition(0, x=90,  y=300)
    .set_context(target=chain_mid_r)
        .SetPosition(0, x=270, y=300)
    .set_context(target=chain_exit)
        .SetPosition(0, x=420, y=420)
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
        .Pulse(0, rgb(50, 70, 230), t=6.1)
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


MOVE_IN_T   = 1.5   # entry offscreen -> mid_l
SWEEP_T     = 3.0   # mid_l -> mid_r (shooting phase)
MOVE_OUT_T  = 1.5   # mid_r -> exit offscreen
WAVES       = 8
WAVE_INT    = SWEEP_T / (WAVES + 1)  # evenly spread across the sweep

from random import randint, random


def make_patrol_enemy(enemy_g: int, bullet_type: lib.BulletPool) -> Component:
    """Build one patrol enemy attack component for the given enemy group."""
    shoot_start = MOVE_IN_T  # waves begin as soon as sweep starts

    comp = (Component(f"PatrolEnemy [{enemy_g}]", unknown_g(), 5)
        .assert_spawn_order(True)

        # -- Enter: snap to entry point, glide to mid_l --
        .set_context(target=enemy_g)
            .GotoGroup(0, chain_entry)
            .GotoGroup(e.TICK, chain_mid_l, t=MOVE_IN_T, type=e.Easing.EASE_IN_OUT, rate=2)
            .Pulse(0, rgb(0, 255, 0), t=MOVE_IN_T + SWEEP_T + MOVE_OUT_T)

        # -- Sweep: glide from mid_l to mid_r while shooting --
            .GotoGroup(MOVE_IN_T, chain_mid_r, t=SWEEP_T, type=e.Easing.NONE)

        # -- Exit: glide to offscreen top-right --
            .GotoGroup(MOVE_IN_T + SWEEP_T, chain_exit, t=MOVE_OUT_T, type=e.Easing.EASE_IN, rate=2)
            .Toggle(MOVE_IN_T + SWEEP_T + MOVE_OUT_T, False)  # despawn at end of exit
        .clear_context()
    )

    # Fire 5 radial waves evenly spaced across the sweep
    comp.pointer.SetPointerCircle(shoot_start, location=enemy_g, follow=True)
    for i in range(WAVES):
        wave_t = shoot_start + (i + 1.5 - random()) * WAVE_INT
        comp.instant.Radial(wave_t, test_bullet, bullet_type, numBullets=24)
    comp.set_context(target=comp.pointer.pc.all)
    comp.Follow(0, enemy_g, t=MOVE_IN_T + SWEEP_T + MOVE_OUT_T)
    comp.pointer.CleanPointerCircle()

    return comp

CHAIN_SIZE   = 6
CHAIN_OFFSET = 1.8   # seconds between each enemy entering

BulletAlloc.start()

chain_enemies: list[tuple[int, Component]] = []
for i in range(CHAIN_SIZE):
    groups = yinyang.next()
    g = groups[0]
    chain_enemies.append((g, make_patrol_enemy(g, lib.bullet1 if i % 2 == 0 else lib.bullet3)))

# ===========================================================================
# SPAWN CALLS
# ===========================================================================

for i, (g, comp) in enumerate(chain_enemies):
    t0 = 1.0 + i * CHAIN_OFFSET
    yinyang.spawn_enemy(Stage.stage1, t0, comp, 10, (g,),
        drops=[(lib.score_pickup, randint(5,13)), (lib.p_pickup, randint(10,20))])


Stage.stage1.Spawn(0, pos_setup.caller, True)

add_enemy_collisions()
add_disable_all_prefabs()
add_plr_collisions()
add_pickup_collisions()

BulletAlloc.resolve()

save_all(level=level, live_editor=False)