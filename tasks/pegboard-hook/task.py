"""On the station table: a white pegboard stands at the far side on pine
furring strips against a grey back panel. Two hooks already sit in its
centre column, each holding a blister-pack card. Two new hooks lie flat
on the table, one in front of each arm. The robot must pick each new
hook up by its arm, tip it up so its curved top prong enters the tagged
hole above its orange tag, turn it level so the prong curls in behind the
board and the bottom prong swings into the hole below, and let go of both
hooks.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import hookgeom as G  # noqa: E402
import numpy as np  # noqa: E402
import scoring  # noqa: E402
from bodykit import Body, BoxShape, MeshShape, on_reset, on_step, rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 0.0  # the score's own 1 s "let go" window is the hold
TOP = TABLE_TOP[SCENE]
CONTROL_HZ = 20

INSTRUCTION = "Install the two new peg hooks in the tagged holes, then let go of both hooks."

MM = 0.001
HERE = pathlib.Path(__file__).resolve().parent

# -- layout (world frame: x forward from the arms, y left, z up, metres; heights above the table top) -----
BOARD_FRONT_X = 0.42
BOARD_W, BOARD_H, BOARD_T = 0.500, 0.350, 0.00635
BOARD_BOTTOM = 0.050  # bottom edge above the table top
BOARD_CENTRE = (BOARD_FRONT_X + BOARD_T / 2, 0.0, TOP + BOARD_BOTTOM + BOARD_H / 2)
HOLE_D, PITCH = 0.0071, 0.0254
HOLE_COLS = tuple(range(-9, 10))  # hole y = m * PITCH
HOLE_ROWS = tuple(range(3, 16))  # hole height above table = n * PITCH
BACK_GAP = 0.020  # gap between the board's back face and the back panel
BACK_PANEL_T = 0.006
STRIP_W = 0.040  # furring strips: width (y) x height (full board height)
STRIP_Y = 0.15

TAG_Y = 0.0762  # tagged holes: A at +TAG_Y, B at -TAG_Y (m = +/-3)
TAG_ROW = 8  # tagged holes and the upper existing hook: 8 * PITCH = 203.2 mm above the table
EXIST_ROWS = (8, 6)  # existing hooks' top holes, centre column (y = 0): 203.2 and 152.4 mm up
TAG_SIZE = (0.0005, 0.015, 0.010)  # orange tags: thickness x width (y) x height (z)
TAG_DROP = 0.012  # tag centre below its tagged hole

# Peg hook: 3/16 in (4.8 mm) wire; the shape is in hookgeom.py (shared with the mesh and the demo).
# Hook frame: origin at the top hole's centre on the board's front face, +x into the board, +z up.
WIRE_R = G.WIRE_R
HOOK_MASS = G.HOOK_MASS
BACK_X = G.BACK_X  # back wire centre, touching the board face
BOTTOM_TIP_X = G.BOTTOM_TIP_X  # bottom prong tip, 8 mm past the front face
PRONG_DROP = G.PRONG_DROP  # bottom prong one pitch below the top prong
TOP_TIP = np.array(G.TOP_TIP)  # curved top prong's tip: 12.6 mm past the face, 2.3 mm above the hole axis
ARM_END_X = G.ARM_END_X
TIP_END = G.TIP_END
NEW_HOOK_POS = ((0.22, 0.08), (0.22, -0.08))  # (x, y) of each new hook's origin, lying flat
CARD_MASS = 0.060
# The two products differ, as on a real display: the upper hook (arm one pitch
# above the lower hook's) can only carry a short pack, or the pack would reach
# the lower hook's arm. (size = thickness x width (y) x height (z), metres;
# hole_off = hang-hole centre above the card centre; on_arm = distance in
# front of the board face where the card hangs.)
CARD_INFO = {
    "card_0": {"size": (0.002, 0.100, 0.045), "hole_off": 0.0145, "on_arm": 0.040, "obj": "card_small.obj"},
    "card_1": {"size": (0.002, 0.120, 0.120), "hole_off": 0.045, "on_arm": 0.080, "obj": "card.obj"},
}

WHITE = (0.94, 0.94, 0.92)
GREY = (0.55, 0.56, 0.58)
PINE = (0.76, 0.62, 0.42)
ORANGE = (1.0, 0.45, 0.05)
STEEL = (0.72, 0.73, 0.75)
CARD_COLOURS = ((0.85, 0.30, 0.25), (0.25, 0.45, 0.80))


def hole_world(y: float, row: int) -> np.ndarray:
    """Centre of a board hole on the front face."""
    return np.array([BOARD_FRONT_X, y, TOP + row * PITCH])


def tag_hole(name: str) -> np.ndarray:
    return hole_world(TAG_Y if name == "a" else -TAG_Y, TAG_ROW)


def _quat_from_matrix(m: np.ndarray):
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


# Lying rotations (columns are the hook's axes in world coordinates). Both hooks lie flat, arms
# toward the robot. Hook A's prongs point to -y and hook B's to +y (mirror images), so each is
# picked up from above by the arm on its own side and needs no regrasp to face the board.
LYING_ROTS = {
    "hook_a": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "hook_b": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}
LYING_ROT = LYING_ROTS["hook_a"]
LYING_XYZWS = {n: tuple(float(v) for v in _quat_from_matrix(r)) for n, r in LYING_ROTS.items()}


def _pegboard() -> Body:
    shapes = [MeshShape(obj=str(HERE / "assets" / "pegboard.obj"), centre=(0.0, 0.0, 0.0),
                        size=(BOARD_T, BOARD_W, BOARD_H), color=WHITE, sdf_resolution=1024)]
    for y in (TAG_Y, -TAG_Y):
        local_z = (TAG_ROW * PITCH - TAG_DROP) - (BOARD_BOTTOM + BOARD_H / 2)
        shapes.append(BoxShape((-BOARD_T / 2 - TAG_SIZE[0] / 2, y, local_z), TAG_SIZE, ORANGE, collide=False))
    return Body.build("pegboard", pos=BOARD_CENTRE, mass=1.0, shapes=shapes, kinematic=True,
                      friction=(0.5, 0.4), color=WHITE, contact_offset=0.0001, solver_iterations=32)


def _back_assembly():
    back_x = BOARD_FRONT_X + BOARD_T + BACK_GAP + BACK_PANEL_T / 2
    panel = Body.build("back_panel", pos=(back_x, 0.0, BOARD_CENTRE[2]), mass=1.0,
                       shapes=[BoxShape((0.0, 0.0, 0.0), (BACK_PANEL_T, BOARD_W, BOARD_H), GREY)],
                       kinematic=True, friction=(0.5, 0.4), color=GREY)
    strip_x = BOARD_FRONT_X + BOARD_T + BACK_GAP / 2
    strips = [Body.build(f"strip_{'a' if s > 0 else 'b'}", pos=(strip_x, s * STRIP_Y, BOARD_CENTRE[2]),
                         mass=1.0, shapes=[BoxShape((0.0, 0.0, 0.0), (BACK_GAP, STRIP_W, BOARD_H), PINE)],
                         kinematic=True, friction=(0.5, 0.4), color=PINE) for s in (1.0, -1.0)]
    return [panel, *strips]


def _hook(name: str, pos, start_xyzw=None) -> Body:
    return Body.build(name, pos=pos, mass=HOOK_MASS,
                      shapes=[MeshShape(obj=str(HERE / "assets" / "hook.obj"), centre=(0.0, 0.0, 0.0),
                                        size=(0.170, 2 * WIRE_R, 0.033), color=STEEL, sdf_resolution=256)],
                      friction=(0.4, 0.3), color=STEEL, contact_offset=0.0002, solver_iterations=32,
                      start_xyzw=start_xyzw)


def _card(name: str, exist_row: int, colour) -> Body:
    info = CARD_INFO[name]
    size = info["size"]
    arm_z = TOP + (exist_row - 1) * PITCH  # the hook's arm is one pitch below its top hole
    pos = (BOARD_FRONT_X - info["on_arm"], 0.0, arm_z - info["hole_off"])
    return Body.build(name, pos=pos, mass=CARD_MASS,
                      shapes=[MeshShape(obj=str(HERE / "assets" / info["obj"]), centre=(0.0, 0.0, 0.0),
                                        size=size, color=colour, sdf_resolution=256),
                              # generic artwork: plain bars, no brands (visual only)
                              BoxShape((-(size[0] / 2 + 0.0002), 0.0, -size[2] * 0.25),
                                       (0.0004, size[1] * 0.65, size[2] * 0.4),
                                       (0.95, 0.95, 0.95), collide=False)],
                      friction=(0.5, 0.4), color=colour, contact_offset=0.0005, solver_iterations=16)


def _objects():
    out = [_pegboard(), *_back_assembly()]
    # New hooks lying flat on the table, arms toward the robot.
    for name, (x, y) in zip(("hook_a", "hook_b"), NEW_HOOK_POS):
        out.append(_hook(name, (x, y, TOP + WIRE_R), start_xyzw=LYING_XYZWS[name]))
    # Existing hooks seated in the centre column, each holding a card.
    for i, row in enumerate(EXIST_ROWS):
        h = hole_world(0.0, row)
        out.append(_hook(f"exist_{i}", tuple(h)))
        out.append(_card(f"card_{i}", row, CARD_COLOURS[i]))
    return out


OBJECTS = _objects()
NEW_HOOKS = ("hook_a", "hook_b")
EXIST_HOOKS = ("exist_0", "exist_1")
CARDS = ("card_0", "card_1")

# -- episode state, followed after every control step --------------------------------------------------------

_CLEAR = scoring.CLEAR_MM * MM


def _fresh() -> dict:
    return {"tcps": None, "clear_steps": 0, "tick": 0, "scored": -1, "score": 0.0}


_EP = _fresh()


@on_reset
def _new_episode() -> None:
    _EP.clear()
    _EP.update(_fresh())


def _tcps(env) -> list[np.ndarray]:
    from bodykit import as_numpy
    from yam_arena.constants import TCP_OFFSET

    base = env.unwrapped
    origin = as_numpy(base.scene.env_origins[0])
    out = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        out.append(g[:3] - origin + rotation(g[3:7]) @ np.array(TCP_OFFSET))
    return out


def _body(env, name):
    from bodykit import _SPECS, _body_state

    return _body_state(env, _SPECS[name])


_HOOK_WIRE_LOCAL = [np.array(p, float) for p in G.wire_polyline()]
_TOP_PRONG_LOCAL = np.array(G.top_prong_points())


def _wire_points(state) -> np.ndarray:
    return np.array([state.pos + state.rotation() @ p for p in _HOOK_WIRE_LOCAL])


def _dist_to_polyline(point, pts) -> float:
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1)))


@on_step
def _follow(env) -> None:
    _EP["tick"] += 1
    try:
        _EP["tcps"] = _tcps(env)
    except Exception:  # noqa: BLE001 - an unknown grip never counts as let go
        _EP["tcps"] = None
    if _EP["tcps"] is None:
        _EP["clear_steps"] = 0
        return
    hooks = [_body(env, n) for n in NEW_HOOKS]
    clear = all(_dist_to_polyline(t, _wire_points(h)) - WIRE_R >= _CLEAR for t in _EP["tcps"] for h in hooks)
    _EP["clear_steps"] = _EP["clear_steps"] + 1 if clear else 0


# -- scoring ----------------------------------------------------------------------------------------------------

def _seg_point_dist(point, a, b) -> float:
    ab = np.asarray(b, float) - np.asarray(a, float)
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(np.asarray(a, float) + t * ab - point))


def hook_measure(objects, name: str, hole: np.ndarray) -> scoring.HookMeasure:
    """One hook measured against ``hole`` (a tagged or existing top hole, front-face centre)."""
    st = objects[name]
    r = st.rotation()
    prong = st.pos + _TOP_PRONG_LOCAL @ r.T  # top prong centreline, world
    tip = st.pos + r @ TOP_TIP
    bottom_tip = st.pos + r @ np.array([BOTTOM_TIP_X, 0.0, -PRONG_DROP])
    back_mid = st.pos + r @ np.array([BACK_X, 0.0, -PRONG_DROP / 2])
    below = hole - np.array([0.0, 0.0, PITCH])
    # Through the hole: the prong's centreline passes the hole's centre, mid-board, within 3.5 mm
    # (the 4.8 mm wire in a 7.1 mm hole can be at most 1.15 mm off it).
    mid_hole = hole + np.array([BOARD_T / 2, 0.0, 0.0])
    top_in = bool(_dist_to_polyline(mid_hole, prong) <= 0.0035)
    tip_behind = bool(tip[0] > BOARD_FRONT_X + BOARD_T)
    # Bottom depth counts only when the bottom tip is lined up with the hole below.
    lined = math.hypot(float(bottom_tip[1] - below[1]), float(bottom_tip[2] - below[2])) <= 0.0035
    bottom_mm = max(0.0, float(bottom_tip[0] - BOARD_FRONT_X) * 1000.0) if lined else 0.0
    back_mm = max(0.0, float(BOARD_FRONT_X - (back_mid[0] + WIRE_R)) * 1000.0)
    arm_dir = r @ np.array([-1.0, 0.0, 0.0])
    out = max(-float(arm_dir[0]), 1e-9)  # how far the arm points away from the board
    from_above = math.degrees(math.atan2(abs(float(arm_dir[1])), out))
    from_side = math.degrees(math.atan2(abs(float(arm_dir[2])), out))
    square = max(from_above, from_side) if arm_dir[0] < 0 else 90.0
    return scoring.HookMeasure(top_in_tagged=top_in, top_tip_behind=tip_behind,
                               bottom_depth_mm=bottom_mm, back_gap_mm=back_mm, arm_square_deg=square)


def _product_hung(objects, card_name: str, hook_name: str) -> bool:
    """The card still hangs on its hook: its hang hole is around that hook's wire (wherever the
    hook sits now; a loaded hook sags a few degrees in its holes) and it hangs upright."""
    st = objects[card_name]
    hole = st.pos + st.rotation() @ np.array([0.0, 0.0, CARD_INFO[card_name]["hole_off"]])
    up = float(st.rotation()[2, 2])
    near = _dist_to_polyline(hole, _wire_points(objects[hook_name])) <= 0.012
    return bool(near and up > math.cos(math.radians(30.0)))


def _existing_in(objects, hook_name: str, row: int) -> bool:
    """An existing hook is still in the board: its top prong through its own hole with the
    curved tip behind the board (it may sag; only coming out counts)."""
    return scoring.is_top_engaged(hook_measure(objects, hook_name, hole_world(0.0, row)))


def end_state(objects) -> scoring.EndState:
    hooks = tuple(hook_measure(objects, n, tag_hole(n[-1])) for n in NEW_HOOKS)
    existing_ok = all(_existing_in(objects, n, r) for n, r in zip(EXIST_HOOKS, EXIST_ROWS))
    products_ok = all(_product_hung(objects, c, h) for c, h in zip(CARDS, EXIST_HOOKS))
    # lowest point of the wire itself (a bounding box would reach below a hook lying on its back)
    off = any(float(_wire_points(objects[n])[:, 2].min()) - WIRE_R < TOP - 0.02 for n in NEW_HOOKS)
    return scoring.EndState(hooks=hooks, clear_s=_EP["clear_steps"] / CONTROL_HZ,
                            product_fallen=not products_ok, existing_knocked=not existing_ok,
                            new_off_table=bool(off))


def _score(objects) -> float:
    if _EP["scored"] != _EP["tick"]:
        _EP.update(scored=_EP["tick"], score=scoring.score(end_state(objects)))
    return _EP["score"]


def success(objects) -> bool:
    """Both new hooks seated at their tagged holes and both grippers clear for the last second."""
    return _score(objects) >= 1.0


def score(objects) -> float:
    """Per new hook 0.15 if only its top prong is engaged in its tagged hole, or 0.4 if seated
    there; plus 0.2 if both are seated and released. 0 if a product or existing hook is
    knocked off, or a new hook is off the table."""
    return _score(objects)
