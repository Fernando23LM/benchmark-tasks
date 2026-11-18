# Install two peg hooks in a store display's pegboard at the tagged holes

A white pegboard stands at the far side of the station table on pine furring strips
against a grey back panel, its face covered with real holes. Two hooks already sit in
its centre column, each holding a blister-pack card. Two new hooks lie flat on the
table, one in front of each arm, each below an orange tag on the board.

The robot has to pick each new hook up by its arm, tip it up in front of the tagged
hole, and turn it down so its curved top prong threads the hole and hooks behind the
board while the same turn swings the bottom prong into the hole below. Then it
presses the back flush and lets go of both hooks. A hook pushed straight in is stopped by its
curved tip; a hook left on its top prong alone sags and has no bottom prong in.

## Why it matters (O*NET)

Fitting peg hooks is how a pegboard display is set up, and stores reset their
pegboard displays with every planogram change. It is listed among the regular
tasks of two occupations that each pass the program's 500,000-worker bar:

- **Stockers and Order Fillers** (O*NET 53-7065.00, about 2.8 million US
  workers): "Design and set up advertising signs and displays of merchandise on
  shelves, counters, or tables to attract customers and promote sales."
  ([O*NET](https://www.onetonline.org/link/summary/53-7065.00))
- **Retail Salespersons** (O*NET 41-2031.00, about 4.0 million US workers):
  "Ticket, arrange, and display merchandise to promote sales."
  ([O*NET](https://www.onetonline.org/link/summary/41-2031.00))

## Run a trial

On a machine with the official simulator (the Robocurve Brev launchable, or
`yam-isaac-lab-arena`'s `setup.sh`), from this folder. `bodykit.py`, `scoring.py`,
`hookgeom.py`, `gen_assets.py` and the meshes in `assets/` must stay next to
`task.py`. Regenerate the meshes after changing any dimension with
`python gen_assets.py`.

```bash
INSTRUCTION="Install the two new peg hooks in the tagged holes, then let go of both hooks."

# Dream-1 (needs a Dream API key in ~/.dreamlabs/credentials)
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$INSTRUCTION" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pegboard-hook-dream
for log in ~/yam-eval/pegboard-hook-dream/*.json; do inspect-robots video "$log"; done

# The scripted demo, which reaches full score
python demo.py task.py demo_out                    # 320x176 cameras
python demo.py task.py demo_out --cam-width 1280   # 1280x720, for a sharper video

# The same demo with no cameras (physics and scoring only; runs on GPUs without RT cores)
python headless.py demo task.py demo_out

# Check the demo's plan against the arm's joint limits without starting the simulator
python demo.py task.py - --dry
```

Both print `task_score` at the end. The demo writes one MP4 per camera, `setup.png`,
`final.png` and `result.json` to its output folder.

The demo is closed-loop, so it has its own runner instead of the `yam_scripted`
policy (which plays fixed waypoints). It drives the same `yam_arena` embodiment
with the same joint-position actions at 20 Hz. It reads where each hook actually
is before each grasp, before the turn, and at every step of the turn, and steers
the arm by the hook's error from the plan (the arm lags its command by up to 12 mm
while it moves). It never moves objects or edits the simulation.
`headless.py` builds the same embodiment without cameras; reset, settling and
scoring are the embodiment's own.

## Scoring

Scored at the end of the trial. `scoring.py` holds every threshold and needs no
simulator. In simulation the 20 N pull is replaced by checking the prongs'
engagement geometry (the curved tip behind the board, the bottom prong at least
5 mm in), **not** by applying a 20 N force after the episode.

A new hook is **seated** at its tagged hole when: its top prong passes through
the tagged hole (its centreline within 3.5 mm of the hole's centre, mid-board) with
the curved tip behind the board's back face; its bottom prong is at least 5 mm
into the hole one pitch below (lined up within 3.5 mm); its back is within 2 mm
of the board face; and its arm is within 10 degrees of square, seen from above
and from the side (each view on its own; a seated 4.8 mm hook sags 5 to 9 degrees
in its 7.1 mm holes).

| Outcome | Score |
|---|---|
| A card off its existing hook (hang hole more than 12 mm from that hook's wire, or the card tilted over 30 degrees), an existing hook out of the board (its top prong out of its hole), or a new hook more than 20 mm below the table top | 0 |
| Per new hook, only its top prong engaged in its tagged hole | 0.15 |
| Per new hook, seated at its tagged hole | 0.4 |
| Both new hooks seated, and both grippers at least 20 mm clear of both new hooks for the last 1 s | +0.2 (1.0 in total) |

On a real bench: the bottom-prong depth and back gap are read from a side photo
with a ruler, the arm angle from a front and a top photo, and a seated hook must
hold a 20 N spring-scale pull along its arm for 1 s.

### Wrong strategies, run in the simulator

`DEMO_VARIANT=<name> python demo.py task.py out` plays these. Scores from
camera-free runs (`headless.py`):

| `DEMO_VARIANT` | What the arms do | Score |
|---|---|---|
| (none) | the full demo | 1.0 |
| `hold` | seats both hooks but keeps hold of the second hook | 0.8 |
| `top_only` | threads both top prongs and lets go part-way through the turn | 0 |
| `one_hook` | installs hook A only | 0.4 |
| `none` | does nothing | 0 |

`top_only` earns nothing because a hook let go part-way through the turn
slides back out of its hole: with only the top prong in, it can't stay. The
0.15 case (a hook left hanging on its top prong at the end) is covered by the
offline tests.

`python -m pytest -q test_scoring.py test_task.py` checks the scoring cases from
the Scoring text (both seated and released 1.0; still held 0.8; one seated and
one top-only 0.55; one seated at an untagged hole 0.4; both top-only 0.3; a back
3 mm off or an arm 12 degrees off not seated, 8 degrees sag still seated; a
fallen card, a pulled-out existing hook, or a new hook on the floor 0; a sagging
existing hook with its card still counts), that the layout, grid, hook and card
geometry in the code match the setup text, and that the curved prong threads
the hole without cutting the board. `test_scoring.py` needs only Python;
`test_task.py` imports the task file, so run it in the simulator's environment.

`python check_fit.py` (no simulator) reports the curved prong's room while it
threads the hole (0.58 mm) and how far a seated hook can sag in the worst case
(5.5 degrees before friction; the simulator settles about 1.5 degrees past that).

## Runtime and seeds

- The scripted demo reaches 1.0 in 1,449 steps of the 2,400 limit (run with
  cameras at 1280x720; 1,448 in the camera-free runs, about 4 minutes on an A100
  including Isaac Sim's start).
- Dream-1 scores 0: in 2,400 steps it reaches both grippers up to the board near
  the tagged holes but never picks up a hook, and leaves the existing hooks and
  products untouched (about 7 minutes on an L40S).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Fidelity

- **Physics:** Isaac Lab-Arena (Isaac Sim, PhysX) rigid bodies at 240 Hz;
  control at 20 Hz.
- **Pegboard:** real size (50 x 35 cm x 6.35 mm) with all 247 real 7.1 mm
  holes (16-sided, 7.1 mm across the flats) on a 25.4 mm grid (19 columns x 13
  rows). The generated mesh (`assets/pegboard.obj`, from `gen_assets.py`) is
  watertight and uses SDF collision at resolution 1024, a 0.1 mm contact offset
  and 32 solver iterations.
- **Hooks:** 3/16 in (4.8 mm) zinc-plated steel wire, 30 g. Shape (`hookgeom.py`,
  shared by the mesh, the scorer and the demo): a back one wire radius in front
  of the board face; a top prong running straight 2 mm into the board, then
  curving up on a 20 mm radius for 28 degrees (tip 11.4 mm behind the face and
  2.3 mm above the hole's axis); a straight bottom prong 8 mm behind the face,
  one pitch below; and a 150 mm arm whose last 25 mm angle up 15 degrees to a
  6 mm ball end. Swept-wire mesh (`assets/hook.obj`) with SDF collision at
  resolution 256, a 0.2 mm contact offset and 32 solver iterations.
  A curve follows itself, so the prong threads the hole when the hook turns
  about the curve's centre; a sharp 90-degree bend can't pass a 7.1 mm hole in
  a 6.35 mm board at any angle. Pulled straight out, the raised tip catches on
  the board. The tip stays well clear of the back panel 20 mm behind the board.
- **Cards:** rigid PET cards (the upper 10 x 4.5 cm, the lower 12 x 12 cm,
  both 2 mm thick, 60 g) with a real 8 mm hang hole (`assets/card.obj`). Generic
  artwork (plain bars), no brands.
- **Assembly:** the board is screwed through two pine furring strips to a
  grey steel back panel; the whole assembly is fixed (kinematic). The
  orange tags are visual only and do not collide.
- **Friction:** hardboard and card 0.5 static / 0.4 dynamic; zinc-plated
  steel wire 0.4 / 0.3 (each body carries one friction pair, combined by
  PhysX averaging).
- **Not modelled:** wire flex (each hook is one rigid body), card flex, tag
  thickness in contact, and any load on the new hooks (no product is hung
  on them during the trial). The 20 N pull is a geometry check in
  simulation, as stated under Scoring.

## Files

- `task.py`: the task file (layout, objects, scoring hooks).
- `scoring.py`: the scoring rules, pure Python.
- `hookgeom.py`: the hook's shape, shared by the mesh, the scorer and the demo.
- `bodykit.py`: rigid bodies built from several shapes and OBJ meshes with
  SDF collision, as an add-on to `yam_arena` task files. It hooks the scene
  builder and the embodiment's step and reset when imported, without
  modifying `yam_arena`.
- `gen_assets.py`: generates `assets/pegboard.obj`, `assets/hook.obj` and
  `assets/card.obj` from the dimensions above.
- `demo.py`: the scripted demo and its wrong-strategy variants.
- `headless.py`: the demo (or `probe_start.py`) with no cameras.
- `check_fit.py`, `check_reach.py`: offline checks of the hook's fit in the
  board and of the arms' reach through the demo.
- `test_scoring.py`, `test_task.py`: offline tests.

## Third-party assets

None. Every mesh is generated in code by `gen_assets.py` from the dimensions
above (a board with real holes, a swept wire hook, and a card with a hang
hole); every other object is built from boxes in code. There are no meshes,
textures, CAD files or artwork from other sources, and no licences to
declare. The scene uses the program's YAM station replica as installed.
