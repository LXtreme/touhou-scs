# Touhou SCS: working guide for coding agents

Updated 2026-11-10 from the working tree and the user's in-game reports.

This project builds **Touhou: Shattered Crystal Shards inside Geometry Dash**.
Python generates GD objects and trigger systems; it does not run inside GD.
The hand-authored level and its prefabs/runtime logic are part of the program.
The long-term six-stage game/story is design intent, not a statement of completion.
The user now expects a shorter game; do not use the six-stage outline as a fixed
scope or treat historical group/object projections as measured current budgets.

Read the relevant source before changing it. The user actively edits alongside
the agent, including during conversations. This guide records contracts and
findings; it must not override newer user instructions or source changes.
`ai_notes.txt` was older background and has since been removed; `remember.txt`
contains ideas and backlog. Historical notes are not authoritative descriptions
of current implementation.

## How the user wants code written

- Use `touhou_scs/component.py` as the style reference: concise methods, distinct
  logical blocks, obvious one-liners where useful, and explicit data flow.
- Keep APIs explicit. A method acts on the target selected by `set_context`.
  Rotation center defaults to that target, assumed to have the correct ID parent.
  Do not silently redirect an operation to an active pointer circle.
- Put pointer-circle invariants in pointer-circle setup, not in unrelated helpers.
- Avoid speculative abstractions, automatic optimization passes, broad validation
  scaffolding, fallback behavior, and arbitrary editor-slider restrictions.
  Retain checks that protect actual implementation requirements.
- Use `obj_id`, `obj_prop`, `obj_prop.Trigger`, and gmdbuilder helper classes.
  Do not write raw `"a123"` property keys in application code when a mapping exists.
  If a mapping is missing, investigate and clearly document any necessary exception.
- Prefer existing builder methods to reconstructing their triggers. For example,
  use `PointToGroup` for Rotate Aim; it now supports an explicit `center=`.
- Prefer `AdvFollow`, `EditAdvFollow`, `ItemCompare`, etc. from `gmdbuilder.classes`
  where applicable. Check their descriptors and serialized fields, not just names
  or docstrings. Some property descriptions are misleading.
- Build an object, assign conditional fields, then append it. Avoid dictionary
  unpacking tricks for optional properties. Avoid unexplained numeric fields.
- Give private state meaningful names and short purpose comments. For trigger
  sequences, explain the operation order in a few words. A helper container called
  "art" was confusing: these objects are invisible position samplers, not artwork.
- Prefer early returns for cache hits to reduce nesting. Keep lazy shared RNG
  storage initialization inside `_get_bit_bank`, returning `(storage, caller)`;
  a separate storage getter/property was unnecessary for this use.
- Preserve builder chaining, language-server completion, and return types when
  moving classes between files. Do not solve typing issues with dynamic attributes.
- Keep readability refactors separate from public API and behavior changes. The
  user deliberately deferred rotation renames, timer keyword renames, and range
  contract changes until their own follow-up work; avoid repeated caller/test churn.
  Preserve explicit `-> Component` annotations on public random helpers. Use an
  explicit `is None` check for an optional center instead of hiding invalid IDs
  with a truthiness fallback.
- Helpers should receive the component they modify explicitly rather than close
  over a particular enemy. A typed `Apply` hook using ParamSpec/Concatenate was
  discussed for fluent local helpers but has not been implemented. Arbitrary
  runtime method injection does not automatically provide static completion.
- Keep optimization tradeoffs explicit. Shared groups reduce stored objects, not
  necessarily runtime operations. Do not report trigger counts as CPU benchmarks.
- Do not commit, send messages, or rewrite unrelated work without instructions.
  Do not invent approval steps for ordinary authorized edits.

## Environment and safe entry points

- Windows/PowerShell workspace; Python executable: `./venv/Scripts/python.exe`.
  Project metadata requires Python >=3.13 and gmdbuilder >=0.2.0.
- `README.md` still mentions G.js/WSLiveEditor. The current generator uses Python,
  gmdbuilder, and `.gdlvl` files; live-editor export also remains supported.
- `main.py` loads `touhou scs.gdlvl` and eventually exports it. Running it is a
  write operation, **not** a harmless smoke test.
- `test.py` and `prefab_prep.py` are executable editor utilities, not unit tests.
  Inspect scripts before running them. `setup_curves.py` rewrites the curve registry
  in `movements.py`; restart the process after changing that registry.
- `pdh.py` is an executable generation utility. The inspected working version
  has no CLI/output guard or fixed generation seed, loads `touhou scs.gdlvl`,
  and calls `save_all(..., live_editor=False)` without binding a new output path.
  Running it normally can overwrite the main source level. Older comparison
  versions had `--rotation`/`--output`; do not assume those options still exist.
  Its current helper replaces the original continuous rotations with random
  phases, a behavior change independent of the library cleanup. PDH refers to
  Junko's **Pristine Danmaku Hell**.
- Preserve user-authored levels and unrelated dirty files. In particular, do not
  overwrite `touhou scs.gdlvl`, `health test.gdlvl`, or a supplied RNG/prototype level
  as a side effect of testing. Use an explicit new output path when authorized to
  generate a diagnostic level. The user has specifically asked to preserve the
  health test during unrelated cleanup.
- Diagnostic scripts/levels are temporary deliverables; clean up only the files
  requested. Several experiments described below were subsequently removed.
- `random_timer_test.py` is a generation utility, not a unit test. It reads
  `rng.gdlvl`, preserves its platformer settings, clears the working object list,
  and exports explicitly to `random_timer_test.gdlvl`. It exports only its timeline
  and RNG storage, avoiding unrelated gameplay/pointer systems. Do not overwrite
  the supplied `rng.gdlvl`. Read actual interval/bits arguments before running:
  the user edits this diagnostic, and its labels/summary can become stale.
- The user uses Zed. Basedpyright is installed through Zed even when it is absent
  from the project venv or shell PATH; see the verification section for its CLI.

### Required Python bootstrap

1. Load/create a gmdbuilder `Level`.
2. Immediately call `touhou_scs.utils.init_level(level)`.
3. Import `touhou_scs.lib` before `touhou_scs.component` in standalone tooling.
4. Construct components/patterns, resolve deferred bullet allocation, then export.

`unknown_g()` delegates to the bound level allocator and skips registered pool
ranges. Import-time definitions allocate groups, so initialization order matters.
`init_level` is one-time per process. `lib` and `component` have an existing circular
dependency; moving another class must not introduce additional import cycles.
Use postponed annotations, `TYPE_CHECKING` imports, and local runtime imports for
helpers that need to construct `Component` or call its module-level utilities.

Use a fresh process for builds. Registries, class caches, allocator binding,
pointer export, and `calltracker` state are not automatically fresh. `calltracker`
sets its called flag in `finally`, including after an exception.

## Architecture and contracts

| File | Responsibility |
| --- | --- |
| `touhou_scs/component.py` | `Component`, validation, contexts, `Pointer`, `Multitarget`, `BulletAlloc`, trigger methods |
| `touhou_scs/comp_extra.py` | `RandomPatterns`, `InstantPatterns`, `TimedPatterns`; exposed by lazy Component properties |
| `touhou_scs/lib.py` | Registries, spells/stages, pools, guider geometry, enemy/drop wiring, export/layout checks |
| `touhou_scs/misc.py` | Collision installation, shot damage/despawn, pickups and shared gameplay behavior |
| `touhou_scs/enums.py` | Game contracts, groups, placeholder IDs, tick duration and constants |
| `touhou_scs/utils.py` | Allocator bootstrap, remap helpers, timing conversions, tracking/validation utilities |
| `touhou_scs/types.py` | Object typing and cross-module protocols |
| `touhou_scs/movements.py` | Bezier fitting/registry, generated movement, optional numerical previews |
| `touhou scs.gdlvl` | Permanent game objects, prefabs, manually authored trigger systems |

### Components, context, and timing

- `Component(name, callerGroup, editorLayer)` registers itself in `lib.all_components`.
  It stores a generated trigger sequence; spawning its caller group runs that sequence.
- `target` is the affected object group. `groups` are execution/gating groups on
  the emitted triggers. Do not conflate them.
- `set_context` persists state; `temp_context` restores it after the block.
  `include_caller=False` is essential for conditional branch groups: otherwise
  their triggers also fire with the whole component.
- Methods usually return the parent Component for fluent chaining. The `.pointer`,
  `.instant`, `.timed`, and `.random` properties lazily provide helper instances.
- A method's `time` is local scheduled time; `t` usually means animation duration.
  `enum.TICK = 1/240`. Time is encoded as X using base speed **311.58 units/sec**.
  It is not a Python delay.
- `assert_spawn_order(True)` preserves relative X timing at export. `False` allows
  layout changes appropriate for untimed groups. This setting is a correctness
  contract, not decoration.
- **Export-order fix:** `_spread_triggers` must preserve logical execution order
  even when exact spacing is not required. Its old all-equal-X/non-ordered branch
  randomized each X independently and sorted afterward, scrambling insertion order.
  That branch now relocates the objects together to one shared X/Y without sorting.
  Rigid layouts preserve spacing and stable X order; mixed-X elastic layouts sort
  by original X first, preserving insertion order for ties, then stretch spacing.
  Do not infer that equal timestamps make ordering irrelevant. Pointer/keyframe
  special layout paths remain separate; these trigger-order guarantees are not a
  general statement about every physical object or GD activation path.
- **Spawn Ordered measures delays from the leftmost trigger in that group.**
  Moving the group's only trigger right does not create an initial wait. A missing
  leftmost event can shift an entire sequence. Use an explicit Spawn delay when
  that is what is intended.
- `enforce_solid_groups` marks physical object groups. Export checks reject treating
  component/trigger groups as objects, with known pool exemptions. Do not bypass
  checks to hide a bad binding.

### Pointers and patterns

- Prebuilt `circle1`: 1080 directions, 1/3-degree spacing; directional groups
  5101–6180, center/all group 6181. Lazy `GuiderCircle`s allocate requested directions.
- `lib.pointer` cycles a **finite pool of 250** generated pointer objects. Reusing
  a pool entry while an earlier system still needs it can corrupt that system.
- `SetPointerCircle` prepares shared geometry at -4/-3/-2 ticks relative to its
  requested time. Its `follow=True` means a common transformation group is created;
  it does not itself mean an Advanced Follow trigger was installed.
- Follow-mode pointer-circle setup registers the center as ID parent of the common
  group in `lib.pointer.parent_registry`. Export applies `obj_prop.PARENT_GROUPS`.
  ID parent, group parent, area parent, and linked group are different concepts.
  `AREA_PARENT` is property 279; do not reintroduce it as a magic string or assume
  it is interchangeable with ID parent. Current pointer export uses ID-parent metadata.
- `CleanPointerCircle` finalizes pointer copying/group membership and clears the
  builder context. It is not a runtime delete. Finish it before export.
- `instant.Arc`/`Radial` use directional pointers and remapped bullet templates;
  `instant.Line` varies simultaneous bullets' travel times. `timed.RadialWave` and
  `timed.Line` schedule repeated emission; `timed.BezierMove` applies relative curves.
- `BulletAlloc` defers supported pattern allocations and resolves them by local
  firing time plus stage offset. It is not a runtime free-list and does not track
  actual bullet lifetimes. Check which pattern methods participate; not all do.
- `BulletPool.next()` cycles a range and returns bullet and collision groups.
  Collision group is offset by pool size. Registered pools also include player
  shots and pickups, so do not treat that registry as hostile bullets only.
- `EnemyPool` can allocate several groups per enemy; preserve the tuple roles.

### Remapping and budgets

- Allocate new IDs through `unknown_g()`. Read `enums.py` and the level contracts
  rather than inventing fixed IDs. GD's group ceiling is 9999.
- Template roles currently include `EMPTY_BULLET=10`, `EMPTY_TARGET_GROUP=20`,
  `EMPTY1=21`, `EMPTY2=23`, `EMPTY_EMITTER=24`, `EMPTY_COLLISION=104`.
  `EMPTY_MULTITARGET=25` is reserved for fan-out machinery. Respect each template's
  required/excluded placeholders. Role names are reused by different templates;
  inspect that template's actual field assignments.
- `Multitarget` shares binary fan-out groups for 1,2,4,8,16,32,64 calls; combined
  requests support up to 127. Larger workloads are split into batches.
- User-provided HDanke explanation: Spawn remap lists are indexed at load; nested
  activations carry a chain of remap indices. At activation, remappable fields are
  transformed through that chain and temporarily applied, then restored. Operations
  which later read the original fields can therefore ignore remapping.
- Remap lists map numeric IDs without distinguishing item/timer/group namespaces,
  even though those namespaces store independent state. Item Edit destinations and
  Random/Item Compare callback IDs are remappable. Choose template roles so a timer
  remap does not accidentally rewrite an unrelated group role in the same path.
- Remap-list length primarily affects size/load; **chain depth and activation volume**
  matter at runtime. Do not assert that arbitrarily nested remaps are free.
  Repeated remapped Spawn loops are a known concern; test the actual chain.
- Grouping reduces stored objects but still executes the routine's internal work.
  Count static objects/groups separately from runtime activations and ongoing actions.
- The latest random-rotation requirement is to **always remap grouped calls** using
  fixed template roles. This supersedes the earlier request to omit identity remaps.
- `lib._enforce_spawn_limit` is a useful static check, not a complete GD simulator.
  Spawn-limit failures depend on callback remap origins and nesting, not only times.

## GD engine findings: distinguish evidence levels

### Timing and callbacks

- **Observed by user:** zero-delay Spawn chains schedule across ticks in the tested
  setup; Item Compare callbacks can traverse a nested chain in one tick. Do not
  substitute Spawn for a synchronous comparison callback and call it zero-tick.
  Some documentation describes Spawn as instant; acknowledge that conflict instead
  of dismissing the user's side-by-side experiment. Do not generalize beyond tested
  activation paths without checking.
- "Zero-tick" numeric calculation/dispatch does not establish that movement or
  rotation has already been physically applied before another action reads it.
- Same timestamp is not proof of a usable result or of safe execution order.
  Separate trigger activation, movement processing, reading position/rotation, and
  projectile emission. Verify the whole dependency chain in game.
- **Documented:** Advanced Follow actions execute sequentially by priority (higher
  first), then activation order for equal priorities. Multiple active actions can
  each apply movement and amplify it. Position updates between actions can matter.
- **Documented:** stopping Advanced Follow is deferred to movement processing.
  Stop/restart in one update can fail to replace the still-active instance.
- A repeated scheduled loop is not automatically single-instance. Before adding
  another start path, inspect loop ownership, pending callbacks, and stop behavior.

### Moves, collisions, and previously fixed bugs

- **User-confirmed fix:** `SetPosition` must schedule its offset at `time + 2*TICK`,
  after the snap to the origin. Multiplying by `2*TICK` scheduled both at zero
  during startup, producing different results in editor and normal play.
  Increasing only the enemy's later read delay did not fix the pointer's internal
  same-tick conflict. Keep snap, offset, and dependent reads as separate steps;
  the confirmed fix retained Silent rather than changing Silent behavior.
- **User-confirmed fix:** silent zero-duration Move when relocating homing bullets
  eliminated erratic trajectories/premature despawns in the Reimu A shot system.
  Component's zero-duration move helpers use Silent. Do not replace this with an
  arbitrary delay or a rotation reset without evidence.
- **User-confirmed improvement:** per-projectile remap origins for enemy-hit
  collision callbacks stopped needles passing through enemies on simultaneous hits.
  A shared installer can be reused, but batching different projectiles under one
  root remap origin caused callback spawn limiting. Preserve that separation.
- Boundary-exit installation is separate from enemy-hit collision installation.
  Initial `onExit` can disable the bullet group and kill other handling if combined
  carelessly. Inspect `misc.add_enemy_collisions` before changing this architecture.
- Live-editor/editor state has historically produced misleading collision behavior.
  Reload/recompile can help, but do not use "GD bug" as a substitute for inspecting
  ordering, callback origins, pooled reuse, toggles, and Silent moves.
- The Reimu shot loop historically uses group 29; groups 156–159 were involved in
  homing stream toggle tests. These are historical investigation landmarks, not
  permission to assume their present definitions.

### Advanced Follow / Edit Advanced Follow

- Mode 1 replaces velocity and is unsuitable for the tested Edit velocity method.
  Mode 2 with acceleration/friction zero can serve as a persistent velocity integrator.
- Edit with `mod_x=mod_y=0` replaces old velocity with its new speed/direction vector;
  those modifiers do not suppress the newly added vector. They clear old momentum.
- **Observed:** tested Edit direction variance behaves as 0..+variance. To get
  `offset +/- spread`, use direction `offset-spread` and variance `2*spread`.
- **Observed:** native Follow activations using Set, and fresh Init activations,
  produced fresh randomized angles. This conflicted with documentation claiming
  coefficients were fixed per object/attempt. Do not treat that documentation as
  settled for every initialization/activation path.
- **Documented:** Mode 2 Rotate Dir follows movement heading, with a rotation-speed
  cap of 0.5 radians (~28.65 degrees) per physics tick. Mode 3 rotation aims at the
  follow center instead. Zero-speed Mode 3 tests did not give usable random rotation.
- `ignore_disabled=True` skips toggled-off targets; it does not stop processing merely
  because an object is offscreen. Hidden artwork is not the same as toggling off.
- For a given Advanced Follow trigger, remapped instances are distinguished by
  target/follow/control IDs. Repeating the same active instance is ignored; a different
  combination is not universally ignored. Do not assume separate Follow trigger
  objects deduplicate merely because they use equal IDs: multiple actions can stack.
  Retarget is only needed when the actual Follow destination must change.
- Important verified emitted fields: Advanced Follow Mode 2=367:1, target=51,
  follow=71, ignore-disabled=336, Rotate Dir=339. Edit speed=300, direction=563,
  direction variance=564, direction reference=565, velocity modifiers=566/568.
  Some explorers previously mislabeled Edit direction as 363/364. Use mappings and
  wrappers in code; these numeric IDs are here to cross-check serialization.

### Current random timer API and caching

```python
component.random.rand_timer(time, minimum, maximum, t=37)  # default: 12 bits
component.random.rand_timer(time, minimum, maximum, t=37, bits=16)
```

- The user replaced the old implementation with a small skeleton and rebuilt only
  random timers so far. `rand_range` and the item-output implementation are gone.
  Timers are sufficient for the user's intended random values; do not restore a
  random-item API or the removed per-destination cache machinery without a request.
  Timer IDs, item IDs, group IDs, and color IDs are distinct namespaces, even if
  their numbers match. `rand_timer` uses `t=`; existing timer builders use `item=`.
- `RandomPatterns.BITS` is now **12**. Bounds must be strictly increasing, and
  supported bit counts are 1-23: the largest raw 23-bit sample is 8,388,607, below
  Item Edit's timer assignment limit of 9,999,999. This check protects the raw-sum
  implementation rather than imposing an arbitrary editor-slider restriction.
- Shared class state consists of `_rng_storage`, `_rand_bits` keyed by **bits**,
  and `_range_entries` keyed by **`(minimum, maximum, bits)`**. A bit bank is the
  Random triggers plus their successful-add branches. An entry contains reset,
  bank dispatch, normalization, range scaling, and optional offset addition.
- A cache hit emits only a Spawn of the existing entry. Output timer, call time,
  component, and target context are not cache keys. Every call supplies the remap
  `{enum.EMPTY1: t}`; all internal arithmetic uses that placeholder as a timer.
  Identical settings share across components and destination timers. There is no
  implicit scratch timer 38 and no bank per destination ID.
- Different offsets, such as 10-20 and 20-30, share their raw bit bank but have
  separate small range entries. The user explicitly chose the full range tuple
  over width-only caching, per-call entry allocation, or an automatic cost-threshold
  optimizer. Keep offset addition separate from bit generation but **inside the
  cached entry**, along with reset and scaling; do not move it outside merely
  because an earlier design discussion considered that option.
- Entry order is: set output to zero; Item Compare `0 == 0` dispatches the shared
  coin flips; divide by `2**bits`; multiply by `maximum - minimum`; add `minimum`
  unless it is zero. Each 50% Random callback adds its power-of-two weight. Branch
  contexts exclude the storage caller, so exporting the container does not make
  those triggers execute unconditionally. Pass helper wrappers' `.obj` to
  `TriggerFromObj`, not the wrapper itself.
- `_rng_storage` uses `assert_spawn_order(False)` and distinct X positions to
  preserve arithmetic order through elastic layout. Entry Spawn is unordered;
  those X gaps express order rather than intended tick delays. Item Compare
  dispatch retains inherited remaps. The user accepts entry latency: **do not
  require or advertise strict zero-tick completion**. Keep calculation together
  behind the Spawn so callers cannot overwrite an intermediate result with a
  loosely scheduled offset edit. Reusing an output timer before a pending roll
  finishes still requires an ordering check.
- Replaying stored routines allocates no new IDs at runtime. A Python loop that
  emits many calls does store one Spawn per call. Repeat enemies/patterns can reuse
  entire routines as well as the RNG banks. For formation-based danmaku, rolling
  once per wave can drive many bullets; do not estimate one roll per bullet by
  default. Historical PDH inspection had 138 rolls for 3,280 bullets. Random boss
  movement/pointer destinations are planned uses, not implemented RNG consumers yet.

### RNG precision and confirmed test findings

- **User-confirmed fix:** Item Edit's **Mod field has three decimal places of
  precision**; that is separate from fractional precision of a computed timer.
  Combining width and normalization into `(maximum-minimum)/2**bits` was wrong.
  For 16 bits, width 10 gives ~0.000153, which becomes zero and leaves a constant
  minimum. Width 100 gives ~0.001526, which becomes 0.002 and permits overshoot.
  Divide by the integer `2**bits` first, then multiply by width in another trigger.
  The original `rng.gdlvl` also normalized by division. Do not reintroduce the
  combined tiny multiplier as an object-saving optimization.
- Constants such as width and minimum still pass through Mod precision. The fix
  does not establish arbitrary precision for bounds below the field's resolution;
  inspect generated constants and their serialized/save behavior when extending
  supported precision. Do not confuse the available output grid with the precision
  of literal trigger fields or the visible counter's formatting.
- **User-confirmed display fix:** a timer counter needs `TIME_COUNTER=True` and
  `SECONDS_ONLY=False` to show decimals. Enabling Seconds Only hid the fractions
  in the first lab. A counter's displayed precision does not establish that the
  stored value is integer or reveal every possible random sample.
- The ideal sample is `minimum + R / 2**bits * (maximum-minimum)`, with integer
  `R` from 0 through `2**bits-1`. Assuming independent fair flips and representable
  constants, the grid is uniform, includes minimum, and excludes maximum. Fewer
  bits mean fewer equally spaced outcomes and more repeats, not a center bias.
  Spacing is `(maximum-minimum)/2**bits`; bits do not specify decimal places.

| Bits | Outcomes | Ideal step across [0, 360) |
| --- | --- | --- |
| 8 | 256 | 1.40625 degrees |
| 10 | 1,024 | 0.3515625 degrees |
| 12 | 4,096 | 0.087890625 degrees |
| 16 | 65,536 | 0.005493164 degrees |

- The user found 10/12 bits visually indistinguishable from 16 in this lab. Twelve
  was recommended for sub-tenth-degree spacing across a full circle and is now the
  source default. Ten may suffice for broad formations/movement. This is a practical
  choice, not measured perceptual equivalence for every bullet pattern.
- The corrected Timer 37 lab was reported working at 0.25-second intervals and at
  **two-tick intervals**. The latter had 600 rolls per range (4,800 total). A one-tick
  version was generated with 1,200 rolls per range (9,600 total); the conversation
  did not establish a separate quantified correctness/performance verdict for it.
  These are numeric timer tests, not proof of physical Rotate/Move readiness.
- The lab lasts 49 seconds: one initial second, then eight five-second rolling
  phases each followed by a one-second hold. Ranges were 10-20, 100-200, 1000-2000,
  100000-200000, 0-1, 0.01-0.02, -20 to -10, and 10.125-10.25. The current diagnostic
  source has since been edited to explicit **10 bits and four-tick intervals**;
  its title/summary still say 16 bits and its subtitle says one tick. Inspect actual
  arguments and interval rather than trusting those labels, or assuming a generated
  level automatically matches a later script edit. Do not regenerate during unrelated
  work just to synchronize a temporary lab.

### RNG object/group costs

These counts describe the current generated structure, excluding labels, test
controls, the caller component's own group, and reserved tagging groups:

| Allocation | Stored triggers | Newly allocated groups |
| --- | --- | --- |
| Shared bank for B bits | 2B | B + 1 |
| Storage container, once across banks | 0 | 1 (unused caller group) |
| First call for a new range tuple, existing bit bank | 6 | 1 |
| Further call for a cached range tuple | 1 | 0 |

- A zero minimum omits the offset trigger: the new-range first call costs 5 instead
  of 6. First use of a bit count also pays for its bank; first use of RNG also pays
  for the storage group. A cold nonzero-minimum 12-bit call costs 30 triggers/15
  groups; the historical 16-bit equivalent costs 38/19.
- For N generated calls, U unique range tuples, and distinct bit counts B, total
  RNG storage is `N + sum(2*B) + sum(4 + (minimum != 0))` triggers and
  `1 + sum(B + 1) + U` allocated groups. The last trigger sum is over unique ranges.
  The 16-bit two-tick lab totaled 4,871 triggers/26 groups, averaging 1.0148 stored
  triggers per call. These are lab-specific amortized counts, not per-roll execution.
- Every roll still activates its outer Spawn, entry, B Random triggers, and the
  successful additions (B/2 on average). Expected activations are `1.5*B + 6` with
  nonzero minimum, or `1.5*B + 5` with zero minimum. Twelve versus sixteen bits saves
  8 stored bank triggers/4 groups and about 6 activations per roll. This does not
  measure CPU time, peak load, or engine scheduling latency.

### Random rotation: pending rebuild and historical findings

- **Rotation is not implemented in the new system yet.** The present `rotate`
  scaffold samples a timer from `_rotate_timers` cycling 60-79 during Python
  generation, but its converter/dispatch lines are commented out. Its docstring
  mentions older timer ranges and caching. Do not treat the method's existence,
  `pdh.py` calling it, or historical passing tests as proof that it currently rotates.
  Random move is also future work. Do not implement either during timer-only tasks.
- The intended direction remains binary rotation with fixed target/center roles
  and remapped grouped calls. The earlier converter cached by target/center/timer
  and used whole-degree flooring; that implementation and per-destination RNG banks
  were removed in this rebuild. Its counts and signatures are historical, not the
  current timer API. Callers explicitly aim/normalize before adding a random delta.
- Preserve pointer-circle ID parenting and `PointToGroup(center=...)`; they remain
  useful independently of the removed random methods. Keep Rotate on each successful
  converter "take" branch when rebuilding; losing that membership caused a past bug.
- Follow/Edit random rotation, its sampler pool, and public `random.Rotate` were
  removed at the user's request. Do not restore them as an optimization. Tests
  showed occasional angle-range violations even at slow rates and progressive
  slowdown in some initialization variants; they did not establish a memory leak.
- Historical binary bullet bursts worked above one tick, while the one-tick shared
  rotation-output test failed. That does not conflict with the new numeric RNG
  results or establish a universal one-tick engine limit. Alternating independent
  rotation outputs remains an untested option. The old `random_rotate_test.py` and
  its stress levels were temporary and removed; the new timer lab is separate.

### Persistent slowdown investigation (unresolved engine cause)

- Early tests suggested lingering lag after bullets despawned. Longer tests with
  960 shots per burst (5,760 total) showed no reported FPS drops in binary or
  repeated-initialization Follow versions. Startup-only Follow initialization
  progressively slowed to zero FPS and an unresponsive game. A first-shot variant
  gated off only the initializer trigger; it progressively slowed and aimed straight
  up. Broken aiming makes it an invalid comparison of correct rotation methods.
- One stored Follow trigger is not evidence of one active runtime action: the
  repeated variant activated it per shot through different outer remap origins.
  Conversely, these observations do not prove that those actions accumulated.
  Do not conclude that all Follow random rotations leak, that repeated activation
  necessarily stacks here, or that a memory leak was measured.
- The inspected hostile-bullet lab reused 700 slots, each with four visual/helper
  objects and one non-dynamic collision block. Moves and pulses were finite (2.5s
  travel); collision listeners were installed once at startup, not per shot.
  These counts describe that lab/pool, not all bullet pools or engine CPU cost.
- Offscreen, invisible, toggled off, and stopped are different states. Toggling off
  a bullet does not cancel every running action; offscreen collision objects can
  still be checked, while inactive ones are skipped. Finite post-despawn actions
  can explain a short lag tail but do not establish indefinite retained work.
- Diagnostic green/blue flashes forced all hostile slots off, then stopped their
  generated actions four seconds later. No clear user report tied FPS recovery
  to either checkpoint. Do not claim that cleanup succeeded or isolated the cause.
  Separate initialization timing, remap context, trigger gating, and cleanup when
  designing another experiment. The user chose binary rotation and removed the
  Follow-random implementation; preserve that decision.

### HP autoscaling lessons

- The prototype uses timer-valued Max HP and Current HP. Different bosses should
  share the same bar logic without assuming a fixed max. Percent of max and percent
  removed are complementary; applying one as the other reverses bar behavior.
- Distinguish a desired absolute bar position from a relative scale/move delta.
  Reset both physical position and tracked position/state when restoring full HP.
- The prototype's full range was 300 small steps; binary moves included 32 and 64.
  An endpoint/flooring issue left 299/300 and a thin sliver at zero HP. Check exact
  empty/full endpoints explicitly, not only representative percentages.
- A direct known-full move on reset was preferred over making the binary update
  system catch up. Remove redundant update dispatches if they serve no purpose.
- Initial Spawn/Item Edit X order mattered in the prototype. Loop start ownership
  must be explicit; avoid accidentally starting independent update loops on each reset.
- Use the 240-TPS tick (~0.004167 s), not ~0.0167 s, for intended one-tick loops.
- These are historical design lessons. Inspect `health test.gdlvl` before asserting
  current item/group assignments; it is a separate prototype and should stay unchanged
  during unrelated work.

## Editing and inspecting GD levels

- Prefer gmdbuilder for production generation and validation; use gmdkit for reading
  raw objects, retaining metadata/settings, and experimental fields beyond wrapper limits.
- Inspect installed `venv/Lib/site-packages/gmdbuilder/classes/`, mappings, enums,
  object types, and gmdkit property descriptors when implementing a trigger. Compare
  emitted fields independently of the wrapper used to generate them.
- `new_obj` / helper `.obj` objects participate in gmdbuilder validation. Bad defaults
  can affect behavior even when every field you explicitly set looks right. Inspect
  defaults, especially when adding a new wrapper or Keyframe behavior.
- `Level.from_file` in gmdbuilder removes generated tag-group objects (normally 9999).
  This is appropriate for rebuilding generated content, but affects object-count
  comparisons and attempts to preserve an existing level exactly.
- `export_to_file()` without an explicit path can target the loaded source. For a
  diagnostic copy, always pass/bind an explicit destination.
- gmdkit `Level.from_file` exposes `.objects`; `.to_file(str(path))` saves explicitly.
  To create a clean platformer lab, copy source metadata/settings, clear objects,
  then generate the new test objects. Do not silently alter unrelated level settings.
- `.gdlvl` files encountered here use URL-safe base64 + gzip around GD XML/plist data;
  key `k4` contains the compressed level string. Inner objects are semicolon-separated
  key/value strings with comma-separated properties. Detect the actual format first.
- GD XML can use short tags (`k`, `s`, etc.) inside `dict`/other variants. Python
  `plistlib` did not reliably read this dialect; ElementTree key/value traversal did.
- Text object **914**, property **31** is base64-encoded UTF-8 text. Use mappings where
  available. Scale X/Y are 128/129. The user liked small, concise labels beside each
  independent prototype with expected behavior, counters, and phase indicators.
  Do not add unnecessary Link Visible/Alpha infrastructure just to display text.
- For side-by-side labs, isolate group/item/control IDs per station, keep source
  settings, and show bounds, center, reference and output clearly. For rotation,
  center the reference artwork when comparing heading. Include a normal-speed build
  whenever slow motion could alter the tested physics.
- Report what was statically inspected or simulated and what was actually observed
  in GD. Do not claim Python tests or generated levels were played in the engine.

## Verification and keeping this guide current

```powershell
.\venv\Scripts\python.exe -m pytest test_random_timer.py test_triggers.py
& "$env:LOCALAPPDATA\Zed\languages\basedpyright\node_modules\.bin\basedpyright.cmd" touhou_scs/comp_extra.py test_triggers.py test_random_timer.py random_timer_test.py
```

The obsolete `test_triggers.py::TestRandomPatterns` and its callback simulator were
removed along with redundant tests; no exclusion is needed. Current RNG tests live
in `test_random_timer.py`; layout regressions are in
`test_triggers.py::TestTriggerLayout`. After cleanup, both files passed together
with 176 tests and no basedpyright errors in those two files.

Basedpyright is installed by the user's Zed setup. The command above was located
and its version checked successfully (1.40.2 on 2026-11-03). If Zed updates its
layout, look under `%LOCALAPPDATA%\Zed\languages\basedpyright` and inspect the
installed package entry points. The venv's `python -m basedpyright` may fail because
that particular Python environment lacks the package; that does not mean the
checker is unavailable. No installation is needed when the Zed CLI can be used.
`pyproject.toml` configures strict typing. Run relevant tests for behavior changes;
a documentation-only change does not require rebuilding levels.

- `conftest.py` initializes a fake allocator and imports lib first. Unit tests do
  not need a live editor. Reset global caches/calltracker state when testing new builds;
  source changes to cache lifetimes can invalidate test-isolation assumptions.
- Callback-graph tests exercise numeric RNG/conversion, not AF physics or GD scheduling.
  Do not write a mock engine and treat its predictions as independent engine evidence.
- Test remap bindings, context restoration, parent export, template sharing
  and independent serialized property IDs where those are real risks.
- A dry validation can build objects and append them to an in-memory gmdbuilder Level
  without calling any export function. This checks schema/layout, not visual behavior.
- For executable utilities, a fresh-process review harness has also replaced
  `Level.export_to_file` with an inspection-only function before `runpy.run_path`.
  That validates the generation/export preparation path without writing a level;
  verify the source hash and distinguish this from a real file export or GD playtest.
- Historical verification after the API reversions: 170 tests passed and PDH passed
  in-memory export validation. Later API edits invalidate that result as a claim
  about today's tree. An isolated converter check covered 2,881 quarter-degree
  values from -360 through +360; it checked arithmetic, not physical Rotate timing.
- Historically, after explicit RNG destinations were implemented on 2026-11-07,
  178 tests passed and Zed's basedpyright reported no errors. Tests cover timer-cycle
  wrap, timer-aware converter caching, same-numbered item/timer independence, and
  integer-range scaling.
  PDH passed an in-memory build/export check with 20 RNG destinations and 100 converter
  banks. That implementation was subsequently removed; those checks do not validate
  the new timer generator or establish in-game scheduling/performance.
- The last recorded checks during the timer rebuild had **187 passed, 14 deselected**
  and no basedpyright errors in `comp_extra.py`, `test_random_timer.py`, and the lab
  generator. Tests cover entry sharing across components/timers, exact cache keys,
  branch membership, timer field types, context preservation, separate division and
  scaling after three-decimal modifier quantization, and trigger layout order.
  Later default-bit/diagnostic edits are not covered by that historical verdict.
  Reset all three RNG class states together for isolated builds/tests:
  `_rng_storage`, `_rand_bits`, and `_range_entries`.
- The old pure-arithmetic tests accepted a tiny multiplier whose precision was lost
  in GD's Mod field. Schema-valid floats and a mathematically correct Python
  formula do not prove literal trigger precision survives the engine. Inspect actual
  serialized fields, model documented precision loss in narrow arithmetic tests,
  and keep the user's playtest results as the evidence for in-game behavior.
- Keep working-tree changes that belong to the user. If tests fail after concurrent
  API edits, inspect current signatures rather than reverting their changes to fit tests.
- Review both staged and unstaged changes before recommending a commit. Reverting
  a file in the working tree does not update its staged version: `MM` can leave old
  keyword arguments staged against restored signatures. Passing working-tree tests
  does not validate a different staged snapshot. Leave staging alone unless asked.
- Keep this file useful: update changed contracts and separate confirmed observations,
  documented behavior, current implementation, and hypotheses. Do not turn theories
  into mandatory engine rules or old prototype IDs into permanent API contracts.

## Research references

Prefer primary documentation and installed mappings, while checking conflicts against
the user's engine observations. The user strongly recommends HDanke's documentation.

- [HDanke GD editor docs](https://uhdanke.github.io/gd_docs/)
- [Advanced Follow](https://uhdanke.github.io/gd_docs/triggers/follows/advanced_follow)
- [Edit Advanced Follow](https://uhdanke.github.io/gd_docs/triggers/follows/edit_advanced_follow)
- [Spawn trigger / Spawn Ordered](https://uhdanke.github.io/gd_docs/triggers/spawn/spawn_trigger)
- [Spawn remapping](https://uhdanke.github.io/gd_docs/triggers/spawn/spawn_remapping)
- [Item Edit](https://uhdanke.github.io/gd_docs/triggers/items/item_edit)
- [Counter labels](https://uhdanke.github.io/gd_docs/triggers/items/counter_labels)
- [G.js algebra precision workaround](https://github.com/g-js-api/algebra#example)
- [Advanced Follow bugs](https://uhdanke.github.io/gd_docs/bugs/advanced_follow)
- [Official GD editor guide](https://robtopgames.com/files/GDEditor.pdf)
- [Flowvix property explorer](https://flowvix.github.io/gd-info-explorer/props)
- [WylieMaster GD documentation](https://wyliemaster.github.io/gddocs/#/)
- [boomlings.dev](https://boomlings.dev/)
- Geode bindings can cross-check engine field names. Library of Geometria and the
  unofficial GD wiki are additional leads, not substitutes for emitted-field checks.
