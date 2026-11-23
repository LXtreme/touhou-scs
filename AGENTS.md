# Touhou SCS: working guide for coding agents

Updated 2026-11-22.

Python generates Geometry Dash triggers. Hand-authored level objects and their
runtime behavior are part of the program. Read the current source and level
before relying on historical notes. The user edits alongside the agent.
`remember.txt` is a backlog, and its six-stage outline and budget estimates
are planning history.

## Project conventions

- Use `component.py` as the style reference. Keep fluent builder APIs typed and
  preserve chaining. Keep API changes separate from readability refactors.
- Respect the caller's target context. A rotation may default its pivot to the
  target, but should never silently redirect to the active pointer circle.
  Keep pointer parenting and setup invariants in the pointer builder.
- Prefer gmdbuilder's mapped fields, wrappers, and existing Component methods.
  Inspect defaults and serialized fields when behavior matters. Wrapper
  descriptions and property explorers can mislabel fields. Pass a wrapper's
  `.obj` to `TriggerFromObj`.
- Share expensive trigger machinery in the library. Let attacks explicitly
  assemble and reuse their own aim/reset/reroll sequences when volume warrants
  it. Avoid automatic cost optimizers, implicit aiming, and arbitrary editor
  slider limits.
- Use `unknown_g()` for generated groups. It skips reserved pool ranges.
  Fixed placeholders are roles within a template, so check inherited remaps
  before reusing a role in a nested call.

## Build and verify safely

- Use the workspace venv in PowerShell. For standalone generation, load a
  gmdbuilder level, call `init_level(level)`, then import `touhou_scs.lib`
  before `touhou_scs.component`. Imports allocate groups and the modules
  depend on each other. Use a fresh process for each build. Allocator binding,
  registries, class caches and pointer export state persist within a process.
  A `calltracker` call can mark itself as called even when it raises.
- `main.py` and `pdh.py` can write back to `touhou scs.gdlvl`. `test.py` and
  `prefab_prep.py` are editor utilities. For a dry export, intercept
  `Level.export_to_file` in a fresh process and validate the generated objects
  in memory. `setup_curves.py` rewrites the curve registry in `movements.py`.
- Preserve the main level, `health test.gdlvl`, and supplied prototypes such
  as `rng.gdlvl`. The random timer and rotation diagnostic scripts use the
  prototype's platformer settings and write to explicitly named outputs.
  Inspect the script and the exported file separately. Diagnostic labels can
  lag behind edits to the test parameters.
- gmdbuilder strips generated tag-group objects (normally 9999) on load.
  Use gmdkit for raw object inspection and when preserving exact source
  objects matters. For raw decoding, these files use URL-safe base64 and gzip
  around GD XML, with the level string in `k4`. GD's short XML tags are not
  reliably parsed by `plistlib`.
- For engine investigations, record actual exported fields and distinguish
  arithmetic or layout checks from playtest evidence. A valid level export
  does not establish when a physical action becomes observable. If behavior
  differs in the editor and in normal play, test from the level start and
  inspect activation order, remap origins, and object lifetimes.

```powershell
.\venv\Scripts\python.exe -m pytest test_random_timer.py test_triggers.py
& "$env:LOCALAPPDATA\Zed\languages\basedpyright\node_modules\.bin\basedpyright.cmd" touhou_scs/comp_extra.py test_random_timer.py test_triggers.py
```

Basedpyright is installed through Zed even when absent from the venv or PATH.
When isolating RNG tests, reset shared trigger storage, bit banks, range
entries, rotation entries, and the converter together. For export tests, also isolate component registries
and pointer export state as applicable. Check both index and working tree
when reviewing a commit. A passing working-tree test does not validate a
different staged snapshot.

## Scheduling, remaps, and object lifetime

- GD runs at 240 ticks/sec. At this project's base speed of 311.58 X units/sec,
  one tick is **1.29825 X units**. Spawn Ordered measures from the leftmost
  trigger in the called group. Anchor the timeline at zero or use explicit
  Spawn delay if the first action is supposed to wait.
- Preserve relative X with `assert_spawn_order(True)` for timed sequences.
  In untimed callback graphs, use distinct X positions to express operation
  order without creating internal waits. The exporter keeps all-equal-X
  untimed triggers together in insertion order. Elastic layouts can separate
  equal-X entries in mixed-X components, keeping their insertion order.
  Layout order alone does not guarantee same-tick engine behavior.
- For branch-only triggers, use `temp_context(..., include_caller=False)`
  to avoid accidentally running the branch with its enclosing component.
  In tested paths, Item Compare callbacks can traverse nested groups in one
  tick, while a zero-delay Spawn can schedule work across ticks. Treat trigger
  dispatch, movement application, and a later heading or position read as
  separate steps.
- Timers, items, groups, and colors have separate state, but Spawn remaps
  operate on numeric IDs. Callbacks can inherit remap chains. Some engine
  operations read a field after its temporary remap has been restored.
  Check each relevant field and nested activation path. Chain depth and
  activation volume matter even when storage is shared.
- The user confirmed that separate per-projectile remap origins prevent
  dropped simultaneous enemy-hit callbacks. A single shared root can hit
  callback limits even with different projectile IDs. Keep boundary-exit
  handlers separate from enemy-hit installation. Initial `onExit` can disable
  bullets before other handlers run. `_enforce_spawn_limit` catches known
  static cases. Playtest nested and simultaneous calls.
- Follow-enabled pointer circles use **ID parents** for their pivots
  (`PARENT_GROUPS`). Area parent, group parent, and linked group are distinct.
  A reused pointer can corrupt a previous consumer that still needs its
  transform. Deferred bullet allocation orders slots by schedule and does
  not track the life of a bullet in GD.
- The confirmed `SetPosition` fix schedules its relative offset at
  `time + 2*TICK` after the initial snap. Delaying only the consumer left
  the pointer itself wrong in normal play. Silent zero-duration moves also
  fixed premature despawns and erratic motion when relocating homing bullets.
- Offscreen, hidden, toggled off, and stopped are different states. Toggling
  off a group can leave actions running. When diagnosing FPS loss, isolate
  initialization, callback origins, live actions, and cleanup before drawing
  conclusions about memory leaks.

## Random timers and binary rotation

- Use timer output for random values. Default 12-bit sampling gives a
  nominal step of about 0.088 degrees over 360 degrees. The user found
  10- and 12-bit samples visually similar to 16-bit in the timer lab.
  Output resolution depends on range width as well as bit count and is
  separate from literal field precision and display formatting.
- **Keep normalization separate from range scaling.** Item Edit's Mod field
  has three decimal places of precision. A merged width/65536 multiplier
  rounded to zero on narrow ranges and rounded up enough to overshoot some
  wider ranges. Dividing by an integer first fixed both in play. Use
  `TIME_COUNTER=True` and `SECONDS_ONLY=False` to show fractional values.
- Keep one raw bit bank per bit count, and cache range and rotation entries
  by `(minimum, maximum, bits)`. Targets, pivots, timer IDs and cadence are
  caller bindings, so they should not multiply the cache. Rotation calls
  share one converter. Each cached call emits one remapped Spawn, with RNG
  and conversion ordered behind it through callbacks. Preserve that order
  if introducing a consumer of the sample.
- The converter consumes its timer and applies the delta rounded down to
  tenths of a degree. In the diagnostic, an Item Edit copies the raw sample
  after RNG and before conversion into a separate display timer. Use an
  explicit timer for a random value that must outlive the scratch cycle.
  Keep long-lived timers out of the scratch range (60-79). Scratch IDs are
  selected per component during generation, so different
  components can reuse the same ID at runtime.
- Use explicit aim/reset calls if an absolute heading is needed. High-volume
  spells can reuse an ordered, branch-only aim/reroll group at the call site.
  Keep the distinction between stored objects and work per activation.
  A cached Spawn adds one trigger and no group per generated call, but still
  executes RNG and conversion.

### Verified heading readback timing

The two-needle lab reset the main needle downward, called random rotation
one tick later, and copied the main heading with a **zero-duration Rotate
in Follow Mode**, Dynamic off. The sampled arc occupied +20 to +80 degrees
clockwise from down, so a downward reading was recognizable as the reset.
The copy lag was measured from the **random-rotate Spawn's X**, not from
the aim reset or the physical rotation.

| Reroll cadence | Copy after 1 tick (1.29825 X) | Copy after 2 ticks (2.59650 X) |
| --- | --- | --- |
| 5 ticks | Often copied reset. Final headings differed. | Worked. Final headings matched. |
| 4 ticks | Mostly copied reset. Final headings differed. | Worked. Final headings matched. |
| 3 ticks | Mostly copied reset. Final headings differed. | Final headings matched. Main needle appeared to leave the arc during the burst. |
| 2 ticks | Both needles unreliable. Final headings differed. | Both needles unreliable during the burst. Final headings matched. |

- **Use 4+ ticks between resets and a read at least 2 ticks after the roll
  Spawn** as the verified schedule for this repeated sequence. In local
  ticks that is aim at 0, roll at 1, read at 3, next aim at 4 or later.
  At a three-tick cadence the late read coincides with the next reset.
  At two ticks even the early read coincides with the next reset.
- End-of-wave agreement is the cleanest readback check because the final
  roll has no following reset. Two ticks matched on every tested final roll.
  This establishes the shortest successful gap tested for that copy path.
  Downward flicker can be a reset read, and the observations do not establish
  whether individual binary Rotate steps are ever separately visible.
- The standalone timer lab worked at two-tick intervals. Physical heading
  application and the eventual bullet consumer still need their own tests.
  For heading labs, reuse `enum.POINTER_OBJ_ID` as the marker and arc dots,
  put the reset outside the sampled arc, and label the copy lag relative to
  the Spawn. Preserve platformer settings and test at normal speed.

## Other engine findings

- For adapting Advanced Follow, Mode 1 replaces velocity and was unsuitable
  for the Edit-velocity method. Mode 2 with no acceleration or friction
  can integrate velocity. Edit with zero X/Y modifiers clears old momentum
  while applying the new vector. Tested direction variance is
  `0..+variance`, so use `offset-spread` and `2*spread` for symmetric jitter.
  Native Set and fresh Init activations produced new random angles in tests,
  despite documentation describing coefficients as fixed for some paths.
- Advanced Follow priorities run high to low, then by activation order
  for ties. Multiple active actions can stack. Stop is deferred to movement
  processing, so stop/restart in the same update may retain the old action.
  Remapped target/follow/control IDs can distinguish active instances.
  Separate trigger objects need not deduplicate. Check position and active
  instance state when retargeting.
- Mode 2 Rotate Dir is rate limited (0.5 rad/tick per the documented
  behavior). Mode 3 points toward the follow center. `ignore_disabled` skips
  toggled-off objects, not simply offscreen or visually hidden objects.
- The user chose binary rotation after Advanced Follow/Edit random rotation
  showed occasional out-of-range angles. Repeated initialization performed
  acceptably in long tests, while some startup-only and gated variants
  slowed down severely. One gated variant also aimed incorrectly. The
  cause of the persistent slowdown remains unresolved. Preserve the
  distinction between that removed randomization technique and Rotate
  Follow Mode, which copies an existing orientation.
- For HP autoscaling, reset both the visible bar and tracked value. Verify
  exact full and empty endpoints in game. Previously, flooring left a sliver
  at zero HP. Establish a single owner for timer-loop startup after resets.

## References

- [HDanke's GD editor docs](https://uhdanke.github.io/gd_docs/)
- [Spawn Ordered](https://uhdanke.github.io/gd_docs/triggers/spawn/spawn_trigger)
- [Spawn remapping](https://uhdanke.github.io/gd_docs/triggers/spawn/spawn_remapping)
- [Item Edit](https://uhdanke.github.io/gd_docs/triggers/items/item_edit)
- [Item Edit modifier precision workaround](https://github.com/g-js-api/algebra#example)
- [Advanced Follow](https://uhdanke.github.io/gd_docs/triggers/follows/advanced_follow)
- [Edit Advanced Follow](https://uhdanke.github.io/gd_docs/triggers/follows/edit_advanced_follow)
- [Official GD editor guide](https://www.robtopgames.com/files/GDEditor.pdf)
- [Geode bindings](https://github.com/geode-sdk/bindings) for cross-checking field names.
