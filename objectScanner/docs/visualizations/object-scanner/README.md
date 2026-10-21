# objectScanner: two streams in motion

Open `index.html` directly in a browser. It is self-contained, makes no network
requests, and starts no camera, capture, tracking or training process.

Every processing step has a distinct 12-second explanatory animation. Select a
step, then use Pause/Play, Replay or the progress slider to inspect its phases.
Flat view removes perspective for easier reading; it is the default on phones
and in the deep reader. With reduced motion enabled, playback stays static and
Next frame advances discretely. Offscreen and background-tab players pause.
An optional stage tour spends 13.5 seconds on each overview step and reveals the
next stream when necessary. Reading or operating a player stops the stage tour.

## Motion concept and parts plan

A physical inspection console displays the selected data transformation on its
glass plane. The front indicator, screen, phase and readout share deterministic
progress. This is an illustration, not live telemetry, literal hardware or a
simulation of measured solver performance.

The console has 41 solids: four feet, a base and cover, four bezel rails, glass,
eight cooling fins, two connector carriers with twelve contacts, four keys, a
front rail, two stops and an indicator. The same geometry serves two overview
players and a shared deep-dive player. Dark/light themes use the same geometry.
`geometry.mjs` retains the earlier bench illustration but is not used by the
current build.

The dataset sequence shows calibrated tag geometry, matched RGB-D capture,
foreground refinement, paired labels, TSDF fusion and two-model training with
pinned geometry. Runtime shows packet alignment, instance/corner association,
cloud formation, PnP or accepted raw history, a schematic rigid GICP update,
and accepted/rejected/stale/lost/recovery states.

The TSDF reference is a sibling artifact of training. PnP runs for initialization
and recovery; accepted raw GICP poses seed established tracking. Accepted poses
alone update the display filter. Illustrative inputs and readouts never claim
real accuracy, latency, convergence or training progress.

## Processing deep dives

All twelve lessons contain algorithm substeps, equations, worked examples,
scoped configuration, checks, failure modes and implementation references.
Stage selection synchronizes the overview and deep player. Direct fragments
such as `index.html#pose-04` open the PnP/seed lesson. The overview tour does not
change the lesson being read.

Content lives in `deep-dataset.mjs` and `deep-pose.mjs`; `deep-view.mjs` embeds
all lessons. Source links resolve inside this repository; lesson text and all
interactions remain available when the HTML is copied elsewhere. Collection
and runtime depth defaults are scoped separately. The TSDF and GICP lessons
also link to official Open3D 0.19 references.

## Build and verification

From this directory:

```sh
node build.mjs --audit
node build.mjs --light
node validate-content.mjs
```

The MIT-licensed Anatomy kit is vendored. `motion-instrument.mjs` owns console
geometry, `motion-{dataset,pose}.mjs` own deterministic screen models,
`motion-view.mjs` creates controls, and `motion-live.js` patches SVG state and
manages playback. `live.js` coordinates stage/lesson navigation.

Verified checks:

- Geometry audit: zero issues for 41 solids at five indicator positions.
- Both themes: zero floating ends among 48 checked physical texture endpoints.
- Content: 12 lessons and 53 valid local source references.
- Motion: all 12 steps have at least four distinct states; 60 scrubbed browser
  states synchronize phase, progress, selected lesson and overview selection.
- Browser: timed advancement, pause stability, replay, keyboard seeking,
  offscreen pause, reduced-motion stepping, flat/console views and themes pass.
- Desktop and 390px phone layouts have no page overflow; console is clean.
- Frame sheets show four inspected states for each of the twelve animations.
- Independent source reviews found no material geometry/state/content defects.
  These checks do not validate live hardware or solver performance.

Replay `qa-motion.json`, `qa-motion-reduced.json` and `qa-deep.json` with
Anatomy's `scripts/drive.mjs`, from the project root. Use `--height 1500` for
desktop checks, and `--width 390 --height 1000 --reduced` for reduced-motion
phone checks. `shots/` contains inspected outputs; older bench captures are
historical.

## Arm Lab publication

Live study: https://arm.gradual-lab.dev/research/object-pose

The Arm Lab homepage links this as Research 04. Full-window light/dark editions
are under `/experiments/object-pose/`. `export-web.mjs DESTINATION` creates the
portable bundle and rewrites implementation links to 36 read-only HTML snapshots
containing cited source excerpts and the two linked workflow guides. No entire
repository, datasets, model weights or environment files are published.

The 2026-10-15 release is `/var/www/armlab/releases/20261015T083407Z-object-pose`.
Previous release for rollback: `20261015T040102Z-mycobot-fixed-target`.
Deployment and browser/hash verification evidence is in
`output/deploy_object_pose_20261015/` at the project root. 101 uploaded files were
hash-verified before activation; 58 public files were checked afterward. Public
browser checks cover both layouts, all 12 animations, 36 source links, playback,
themes, existing studies and HTML/RSC responses, with no browser errors.
