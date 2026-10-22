 
from collections.abc import Iterator

from gmdbuilder import from_object_string, obj_id, obj_prop
from gmdbuilder.classes import ItemCompare
from touhou_scs import enums as enum, lib, utils as util
from touhou_scs.component import Component, Multitarget
from touhou_scs.utils import calltracker, unknown_g

# Hitbox is the weapon, hurtbox is the target
BOUNDARY_HITBOX = 1
PLR_HURTBOX = 2
PLR_GRAZE_HURTBOX = 3
BOMB_HITBOX = 4
GRAZE_FUNCTION = 34
PLR_HURT_FUNCTION = 35
DESPAWN_FUNCTION = 27 #PLR_HURT calls despawn in level, BOMB_HURT calls directly in code
ENEMY_HITBOX = 5 # shared for every enemy
GLOBAL_COLLISIONS = 17

ppt = obj_prop.Trigger

@calltracker
def add_disable_all_prefabs():
    if add_disable_all_prefabs.has_been_called:
        raise RuntimeError("Disable All Prefabs has already been added")

    comp = Component("Disable All Prefabs", 32, editorLayer=4) \
        .assert_spawn_order(False)

    single = (Component("Disable Single Prefab", unknown_g(), editorLayer=6)
        .assert_spawn_order(False)
        .set_context(target=enum.EMPTY_BULLET)
            .Toggle(0, False)
        .clear_context()
    )

    prefab_groups = [
        g
        for pool in lib.registered_bullet_pools
        for g in range(pool.min_group, pool.max_group + 1)
    ] + [
        g
        for pool in lib.registered_enemy_pools
        for g in range(pool.min_group, pool.max_group + 1, pool.num_groups_per)
    ]

    prefab_iter = iter(prefab_groups)
    remaining = len(prefab_groups)

    def remap_disable(remap_pairs: dict[int, int], remap: util.Remap):
        for source, target in remap_pairs.items():
            if source == enum.EMPTY_BULLET:
                remap.pair(target, next(prefab_iter))
            else:
                remap.pair(target, enum.EMPTY_MULTITARGET)
    
    # Chunk into max-64 batches and handle remainder normally
    while remaining > 0:
        batch_size = 64 if remaining > 127 else remaining

        Multitarget.spawn_with_remap(comp, 0, batch_size, single, remap_disable)
        remaining -= batch_size


@calltracker
def add_plr_collisions():
    if add_plr_collisions.has_been_called:
        raise RuntimeError("Collisions have already been added")

    cols = (Component("Base Enemy's Bullet Collisions (un-mapped)", 18, editorLayer=6)
        .assert_spawn_order(False)
        .set_context(target=enum.EMPTY_BULLET)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=BOUNDARY_HITBOX, activateGroup=False, onExit=True)
        .set_context(target=GRAZE_FUNCTION)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=PLR_GRAZE_HURTBOX, activateGroup=True)
        .set_context(target=DESPAWN_FUNCTION)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=BOMB_HITBOX, activateGroup=True)
        .set_context(target=enum.EMPTY_BULLET)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=PLR_HURTBOX, activateGroup=True)
        .clear_context()
    )

    placeholder = unknown_g() # never called, just fulfills comp param requirement
    def add_collision_trigger_remaps(bullet: lib.BulletPool, name: str):
        # Called on level startup
        global_col = Component(
            f"[{name}]: Bullet Collision remap wrappers", GLOBAL_COLLISIONS, editorLayer=4) \
            .assert_spawn_order(False)
        # Each trigger in here is called individually, grouped for convenience/logging
        plr_hit_col = Component(f"[{name}]: PlrHit Collisions", placeholder, editorLayer=4) \
            .assert_spawn_order(False)

        for bullet_hitbox in range(bullet.min_group, bullet.max_group + 1):
            bullet_col = bullet_hitbox + (bullet.max_group - bullet.min_group + 1)
            # permanently turns on all collisions for each bullet (level calls it on startup)
            global_col_spawn = global_col.create_trigger(obj_id.Trigger.SPAWN, 0)
            global_col_spawn[ppt.Spawn.TARGET_ID] = cols.caller
            global_col_spawn[ppt.Spawn.REMAPS] = { enum.EMPTY_BULLET: bullet_hitbox }
            # Give each bullet a spawn trigger that activates its own collisions
            with plr_hit_col.temp_context(groups={bullet_hitbox, bullet_col}):
                plr_hit_col_spawn = plr_hit_col.create_trigger(obj_id.Trigger.SPAWN, 0)
                plr_hit_col_spawn[ppt.Spawn.TARGET_ID] = PLR_HURT_FUNCTION
                plr_hit_col_spawn[ppt.Spawn.REMAPS] = { enum.EMPTY_BULLET: bullet_hitbox }

    # DO NOT use the bullet pool registry because that includes pickups
    # TODO: make separate registries for player bullets and pickups
    add_collision_trigger_remaps(lib.bullet1, "B1")
    add_collision_trigger_remaps(lib.bullet2, "B2")
    add_collision_trigger_remaps(lib.bullet3, "B3")
    add_collision_trigger_remaps(lib.bullet4, "B4")

    (Component("Enemy-dmg-Player Collisions", GLOBAL_COLLISIONS, editorLayer=7)
        .assert_spawn_order(False)
        # WARNING: its using empties without remaps
        # (since not remapping PLR_HIT_FUNCTION and therefore not remapping DESPAWN_FUNCTION)
        .set_context(target=PLR_HURT_FUNCTION)
            .Collision(0, blockA=ENEMY_HITBOX, blockB=PLR_HURTBOX, activateGroup=True)
        .clear_context()
        .set_context(target=GRAZE_FUNCTION)
            .Collision(0, blockA=ENEMY_HITBOX, blockB=PLR_GRAZE_HURTBOX, activateGroup=True)
        .clear_context()
    )


@calltracker
def add_enemy_collisions():
    """Does not include enemy damaging the player"""
    if add_enemy_collisions.has_been_called:
        raise RuntimeError("Enemy collision has already been added")

    # Main (index 0) group of every slot across all registered enemy pools
    enemy_groups = [
        g
        for pool in lib.registered_enemy_pools
        for g in range(pool.min_group, pool.max_group + 1, pool.num_groups_per)
    ]

    all_bullet_groups = [
        g for pool, _ in PLAYER_SHOT_TYPES
        for g in range(pool.min_group, pool.max_group + 1)
    ]

    # ── Boundary-exit despawn ─────────────────────────────────────────────────
    # One permanent collision trigger per bullet.  Must be a separate component
    # from the enemy-hit ones — mixing them causes onExit to fire on level load
    # and permanently kill the enemy-hit collision (bullet group gets deactivated
    # before any enemy is ever hit).
    plr_bullet_boundary_col = (Component("PlrBullet Boundary Exit (un-mapped)", unknown_g(), editorLayer=6)
        .assert_spawn_order(False)
        .set_context(target=enum.EMPTY_BULLET)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=BOUNDARY_HITBOX, activateGroup=False, onExit=True)
        .clear_context()
    )
    boundary_spawner = Component("PlrBullet Boundary Exit remap wrappers", GLOBAL_COLLISIONS, editorLayer=4) \
        .assert_spawn_order(False)
    for bg in all_bullet_groups:
        t = boundary_spawner.create_trigger(obj_id.Trigger.SPAWN, 0)
        t[ppt.Spawn.TARGET_ID] = plr_bullet_boundary_col.caller
        t[ppt.Spawn.REMAPS] = {enum.EMPTY_BULLET: bg}

    # ── Enemy-hit collisions ──────────────────────────────────────────────────
    global_col = Component("Enemy Collision remap wrappers", GLOBAL_COLLISIONS, editorLayer=4) \
        .assert_spawn_order(False)

    # One shared base: EMPTY_BULLET hits EMPTY_TARGET_GROUP → activate EMPTY1.
    # EMPTY1 is remapped to the appropriate despawn component per bullet type.
    base_col = (Component("Enemy Collision for PlrBullets (un-mapped)", unknown_g(), editorLayer=6)
        .assert_spawn_order(False)
        .set_context(target=enum.EMPTY1)
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=enum.EMPTY_TARGET_GROUP, activateGroup=True)
        .clear_context()
    )

    def add_bullet_enemy_collisions(bullet_groups: list[int], despawn: Component) -> None:
        """Wire every bullet in bullet_groups to activate despawn when it hits any enemy."""
        for enemy in enemy_groups:
            bullet_iter = iter(bullet_groups)
            remaining   = len(bullet_groups)

            def remap_cb(remap_pairs: dict[int, int], remap: util.Remap,
                         _enemy: int = enemy, _iter: Iterator[int] = bullet_iter):
                for source, target in remap_pairs.items():
                    if source == enum.EMPTY_BULLET:
                        remap.pair(target, next(_iter))
                    elif source == enum.EMPTY_TARGET_GROUP:
                        remap.pair(target, _enemy)
                    elif source == enum.EMPTY1:
                        remap.pair(target, despawn.caller)
                    else:
                        remap.pair(target, enum.EMPTY_MULTITARGET)

            while remaining > 0:
                batch_size = 64 if remaining > 127 else remaining
                Multitarget.spawn_with_remap(global_col, 0, batch_size, base_col, remap_cb)
                remaining -= batch_size

    for pool, despawn in PLAYER_SHOT_TYPES:
        bullet_groups = list(range(pool.min_group, pool.max_group + 1))
        add_bullet_enemy_collisions(bullet_groups, despawn)

    # ── Homing tracker collision ──────────────────────────────────────────────
    homing_func = (Component("Homing Collision remap wrapper", unknown_g(), editorLayer=4)
        .assert_spawn_order(False)
        .set_context(target=enum.HOMING_TRACKER_BLOCK)
            .Toggle(0, False)
        .clear_context()
    )
    retarget = from_object_string(f"1,3661,2,0,20,4,57,{homing_func.caller},62,1,87,1,36,1,51,136,71,20,566,1,568,1;")
    homing_func.triggers.append(retarget) # type: ignore

    base_homing_col = (Component("Enemy Collision for Homing tracker (un-mapped)", unknown_g(), editorLayer=6)
        .assert_spawn_order(False)
        .set_context(target=homing_func.caller)
            .Collision(0, blockA=enum.HOMING_TRACKER_COL_ID, blockB=enum.EMPTY_TARGET_GROUP, activateGroup=True)
        .clear_context()
    )

    enemy_iter = iter(enemy_groups)
    remaining = len(enemy_groups)

    def remap_homing(remap_pairs: dict[int, int], remap: util.Remap):
        for source, target in remap_pairs.items():
            if source == enum.HOMING_TRACKER_COL_ID:
                remap.pair(target, enum.HOMING_TRACKER_COL_ID)
            elif source == enum.EMPTY_TARGET_GROUP:
                remap.pair(target, next(enemy_iter))
            else:
                remap.pair(target, enum.EMPTY_MULTITARGET)

    while remaining > 0:
        batch_size = 64 if remaining > 127 else remaining
        Multitarget.spawn_with_remap(global_col, 0, batch_size, base_homing_col, remap_homing)
        remaining -= batch_size


@calltracker
def add_pickup_collisions():
    """Player colliding with pickups (bombs, powerups, points)"""
    if add_pickup_collisions.has_been_called:
        raise RuntimeError("Pickup collisions have already been added")

    PLR_SCREEN_BOUNDARY = 6 # the player's outer hitbox primarily for boundary collisions

    POWER_STEP = 0.05
    POWER_MAX = 4.00
    SCORE_PER_PICKUP = 100
    TOLERANCE = 0.001 # power is a float timer; absorbs accumulated 0.05-step error
    CROSS_EPS = 0.001 # same idea, baked into both sides of the whole-number compare
    POWER_UP_SFX = "1,3602,62,1,87,1,36,1,392,5920,404,12,406,0.38,421,1,422,0.5,10,0.5,433,1,490,1.148,598,0.01;"
    MAX_POWER_SFX = "1,3602,62,1,87,1,36,1,392,5920,404,8,405,6,406,0.49,421,1,422,0.5,10,0.5,433,1,490,1.447,598,0.01;"

    # compare power >= 4.0
    # ├ true  (already_max_g): score += power * 200
    # └ false (not_max_g): compare 2: (power + 0.05) >= 4.0 (tol 0.001)
    #     ├ true  (reach_max_g): power = 4.0, spawn power_up_logic
    #     └ false (cross_check_g): compare 3: floor(power + eps) < floor(power + 0.05 + eps)
    #         ├ true  (level_up_g): power += 0.05, score += 100, sfx
    #         └ false (plain_add_g): power += 0.05, score += 100
    already_max_g = unknown_g()
    not_max_g = unknown_g()
    reach_max_g = unknown_g()
    cross_check_g = unknown_g()
    level_up_g = unknown_g()
    plain_add_g = unknown_g()
    
    is_full_power = ItemCompare()
    is_full_power.set_from_string(f"t{enum.POWER_LEVEL} >= {POWER_MAX:.2f}")
    is_full_power.true_id = already_max_g
    is_full_power.false_id = not_max_g

    will_be_full_power = ItemCompare()
    will_be_full_power.set_from_string(f"t{enum.POWER_LEVEL} + {POWER_STEP} >= {POWER_MAX:.2f}")
    will_be_full_power.tolerance = TOLERANCE
    will_be_full_power.true_id = reach_max_g
    will_be_full_power.false_id = cross_check_g

    crosses_whole_number = ItemCompare()
    crosses_whole_number.set_from_string(
        f"floor(t{enum.POWER_LEVEL} + {CROSS_EPS:.3f}) < floor(t{enum.POWER_LEVEL} + {POWER_STEP + CROSS_EPS:.3f})")
    crosses_whole_number.true_id = level_up_g
    crosses_whole_number.false_id = plain_add_g

    full_power = (Component("Full Power", unknown_g(), editorLayer=6)
        .assert_spawn_order(True)
        # TODO: power up visuals
        .TriggerFromStr(0, MAX_POWER_SFX)
        .Pickup(0, item_id=enum.SCORE, count=SCORE_PER_PICKUP, override=False)
    )

    on_p_pickup = (Component("On P Pickup Collision", unknown_g(), editorLayer=6)
        .assert_spawn_order(True)
        .set_context(target=enum.EMPTY_BULLET)
            .Scale(0, factor=2, hold=0.1, t=0.3, type=enum.Easing.ELASTIC_IN_OUT, rate=1.5)
            .Alpha(0, t=0.3, opacity=0)
            .Pulse(0, lib.rgb(255,255,255), t=0.3, exclusive=True)
            .Alpha(0.3, t=0, opacity=100)
            .Toggle(0.3, False)
        .clear_context()
        .TriggerFromObj(0, is_full_power.obj)
        .set_context(groups=already_max_g, include_caller=False)
            .Pickup(0, item_id=enum.SCORE, count=SCORE_PER_PICKUP * 2, override=False)
        .set_context(groups=not_max_g, include_caller=False)
            .TriggerFromObj(0, will_be_full_power.obj)
        .set_context(groups=reach_max_g, include_caller=False)
            .TimerOp(0, item=enum.POWER_LEVEL, mod=POWER_MAX) # snap to max
            .Spawn(0, full_power, True)
        .set_context(groups=cross_check_g, include_caller=False)
            .TriggerFromObj(0, crosses_whole_number.obj)
        .set_context(groups=level_up_g, include_caller=False)
            .TimerOp(0, item=enum.POWER_LEVEL, sign=enum.Item.MathOp.ADD, mod=POWER_STEP)
            .Pickup(0, item_id=enum.SCORE, count=SCORE_PER_PICKUP, override=False)
            .TriggerFromStr(0, POWER_UP_SFX)
        .set_context(groups=plain_add_g, include_caller=False)
            .TimerOp(0, item=enum.POWER_LEVEL, sign=enum.Item.MathOp.ADD, mod=POWER_STEP)
            .Pickup(0, item_id=enum.SCORE, count=SCORE_PER_PICKUP, override=False)
        .clear_context()
    )
    
    on_b_pickup = (Component("On B Pickup Collision", unknown_g(), editorLayer=6)
        .assert_spawn_order(True)
        .set_context(target=enum.EMPTY_BULLET)
            .Scale(0, factor=2, hold=0.1, t=0.3, type=enum.Easing.ELASTIC_IN_OUT, rate=1.5)
            .Alpha(0, t=0.3, opacity=0)
            .Pulse(0, lib.rgb(255,255,255), t=0.3, exclusive=True)
            .Alpha(0.3, t=0, opacity=100)
            .Toggle(0.3, False)
        .clear_context()
        .Pickup(0.3, item_id=enum.BOMB_COUNTER, count=1, override=False)
    )

    on_score_pickup = (Component("On Score Pickup Collision", unknown_g(), editorLayer=6)
        .assert_spawn_order(True)
        .set_context(target=enum.EMPTY_BULLET)
            .Scale(0, factor=2, hold=0, t=0.3, type=enum.Easing.ELASTIC_IN_OUT, rate=1.5)
            .Alpha(0, t=0.3, opacity=0)
            .Pulse(0, lib.rgb(255,255,255), t=0.3, exclusive=True)
            .Alpha(0.3, t=0, opacity=100)
            .Toggle(0.3, False)
        .clear_context()
        .Pickup(0.3, item_id=enum.SCORE, count=100, override=False)
    )
    
    def sfx(comp: Component):
        sfx = from_object_string("1,3602,2,405,3,315,155,1,62,1,87,1,36,1,392,4835,404,2,405,16,406,0.8,410,137,411,117,421,1,422,0.5,10,0.5,433,1,490,0.122,502,6,598,0.01;")
        sfx[obj_prop.X] = 0
        sfx[obj_prop.GROUPS] = { comp.caller }
        comp.triggers.append(sfx) # type: ignore
    
    sfx(on_p_pickup)
    sfx(on_b_pickup)
    sfx(on_score_pickup)

    p_group = unknown_g()
    b_group = unknown_g()
    score_group = unknown_g()
    intermediate = unknown_g()

    (Component("Pickup Collisions (unmapped)", 0, editorLayer=6)
        .assert_spawn_order(False)
        .set_context(groups={intermediate})
            .Spawn(0, enum.EMPTY1, True)

        .set_context(target=enum.EMPTY_BULLET, groups={p_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=enum.BOTTOM_BORDER, 
                activateGroup=False, onExit=True)
        .set_context(target=intermediate, groups={p_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=PLR_SCREEN_BOUNDARY, activateGroup=True)

        .set_context(target=enum.EMPTY_BULLET, groups={b_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=enum.BOTTOM_BORDER, 
                activateGroup=False, onExit=True)
        .set_context(target=intermediate, groups={b_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=PLR_SCREEN_BOUNDARY, activateGroup=True)

        .set_context(target=enum.EMPTY_BULLET, groups={score_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=enum.BOTTOM_BORDER, 
                activateGroup=False, onExit=True)
        .set_context(target=intermediate, groups={score_group})
            .Collision(0, blockA=enum.EMPTY_BULLET, blockB=PLR_SCREEN_BOUNDARY, activateGroup=True)
    )
    
    global_col = Component("Pickup Collision remap wrappers", GLOBAL_COLLISIONS, editorLayer=4) \
        .assert_spawn_order(False)

    magnet_col = Component("Pickup Magnet Triggers", unknown_g(), editorLayer=6) \

    # optimization: if replacing with multitarget, make
    for i in range(lib.p_pickup.min_group, lib.p_pickup.max_group + 1):
        global_col.Spawn(0, p_group, False, 
            remap={enum.EMPTY_BULLET: i, enum.EMPTY1: on_p_pickup.caller})
        with magnet_col.temp_context(groups={124}, target=i):
            magnet_col.GotoGroup(0, enum.PLR, t=0.5, dynamic=True, type=1, rate=2)
    for i in range(lib.b_pickup.min_group, lib.b_pickup.max_group + 1):
        global_col.Spawn(0, b_group, False, 
            remap={enum.EMPTY_BULLET: i, enum.EMPTY1: on_b_pickup.caller})
        with magnet_col.temp_context(groups={124}, target=i):
            magnet_col.GotoGroup(0, enum.PLR, t=0.5, dynamic=True, type=1, rate=2)
    for i in range(lib.score_pickup.min_group, lib.score_pickup.max_group + 1):
        global_col.Spawn(0, score_group, False, 
            remap={enum.EMPTY_BULLET: i, enum.EMPTY1: on_score_pickup.caller})
        with magnet_col.temp_context(groups={124}, target=i):
            magnet_col.GotoGroup(0, enum.PLR, t=0.5, dynamic=True, type=1, rate=2)


despawn1 = (Component("EnemyBullet Despawn 1", unknown_g(), editorLayer=6)
    .assert_spawn_order(True)
    .set_context(target=enum.EMPTY_BULLET)
        .Scale(0, factor=0.25, hold=0.01, t=1, type=enum.Easing.ELASTIC_IN_OUT, rate=1.2)
        .Alpha(0, t=1, opacity=0)
        .Pulse(0, lib.rgb(0,0,0), t=1)
        .Alpha(1, t=0, opacity=100)
        .Toggle(1, False)
    .clear_context()
)

despawn2 = (Component("PlrBullet Despawn 1", unknown_g(), editorLayer=6)
    .assert_spawn_order(True)
    # Bullet despawn
    .set_context(target=enum.EMPTY_BULLET)
        .Scale(0, factor=0.25, hold=0.1, t=0.1, type=enum.Easing.ELASTIC_IN_OUT, rate=1.5)
        .Alpha(0, t=0.1, opacity=0)
        .Pulse(0, lib.rgb(0,50,255), t=0.2, exclusive=True)
        .Alpha(0.2, t=0, opacity=100)
        .Toggle(0.2, False)
    .clear_context()
)

def make_shot_despawn(name: str, damage: tuple[float, float, float, float, float], despawn_g: int) -> Component:
    """Create a per-power-level damage + bullet-despawn component for a player shot type."""
    l0, l1, l2, l3, l4 = damage
    return (Component(f"{name} Despawn", unknown_g(), editorLayer=6)
        .assert_spawn_order(False)
        .set_context(groups=enum.PowerLevel.LEVEL_0)
            .TimerOp(0, item=enum.EMPTY_TARGET_GROUP, sign=enum.Item.MathOp.ADD, mod=l0)
        .set_context(groups=enum.PowerLevel.LEVEL_1)
            .TimerOp(0, item=enum.EMPTY_TARGET_GROUP, sign=enum.Item.MathOp.ADD, mod=l1)
        .set_context(groups=enum.PowerLevel.LEVEL_2)
            .TimerOp(0, item=enum.EMPTY_TARGET_GROUP, sign=enum.Item.MathOp.ADD, mod=l2)
        .set_context(groups=enum.PowerLevel.LEVEL_3)
            .TimerOp(0, item=enum.EMPTY_TARGET_GROUP, sign=enum.Item.MathOp.ADD, mod=l3)
        .set_context(groups=enum.PowerLevel.LEVEL_4)
            .TimerOp(0, item=enum.EMPTY_TARGET_GROUP, sign=enum.Item.MathOp.ADD, mod=l4)
        .clear_context()
        .set_context(target=enum.EMPTY_TARGET_GROUP)
            .Pulse(0, lib.rgb(0,0,255), fadeIn=0.1, fadeOut=0.1, exclusive=True)
        .clear_context()
        # player bullet hitting enemy sfx
        .TriggerFromStr(0, "1,3602,2,225,3,-225,62,1,87,1,36,1,392,545,404,5,405,-8,406,0.17,421,1,422,0.5,10,0.5,433,1,490,0.188,597,1;")
        .Spawn(0, despawn_g, True)  
    )

ofuda_bullet_despawn = make_shot_despawn("ReimuA Regular", (-0.4, -0.5, -0.6, -0.7, -0.8), despawn2.caller)
homing_bullet_despawn = make_shot_despawn("ReimuA Homing",  (-0.3, -0.35, -0.4, -0.5, -0.6), despawn2.caller)
bomb_bullet_despawn = make_shot_despawn("ReimuA Bomb", (-1.0, -1.2,  -1.4, -1.6, -2.0), enum.EMPTY_BULLET)
needle_bullet_despawn = make_shot_despawn("ReimuA Needle", (-0.7*1.15, -0.8*1.15, -0.9/2*1.15, -1/2*1.15, -1.2/2*1.15), despawn2.caller)


# =============================
# ENEMY BULLET DESPAWN FUNCTION
# =============================

enemy_bullet_despawn = (Component("EnemyBullet Despawn List", DESPAWN_FUNCTION, editorLayer=6)
    .assert_spawn_order(False)
    # Note: if a collisionX component seems to be be spawning delayed, its a GD bug. reload level.
        .Spawn(0, despawn1.caller, True) # toggle this on/off same tick w/ unique group
)


# Shot types registered for enemy collision and boundary despawn.
# To add a new shot type: add its BulletPool to lib.py, define its despawn
# with make_shot_despawn above, then add one entry here.
PLAYER_SHOT_TYPES: list[tuple[lib.BulletPool, Component]] = [
    (lib.reimuA_level1, ofuda_bullet_despawn),
    (lib.reimuA_homing_shots, homing_bullet_despawn),
    (lib.reimuA_bomb_balls, bomb_bullet_despawn),
    (lib.reimuA_needles, needle_bullet_despawn),
]
