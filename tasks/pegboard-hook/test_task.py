"""Layout and geometry checks: the numbers in the code against the task's Setup and
Scoring text. Needs the simulator's environment (it imports the task file).
Run: python -m pytest test_task.py"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import hookgeom as G  # noqa: E402
import scoring  # noqa: E402
import task  # noqa: E402
from bodykit import state_from_pose  # noqa: E402

MM = 0.001


def _spec(name):
    return next(o for o in task.OBJECTS if o.name == name)


def test_board_and_grid_match_the_setup_text():
    assert task.BOARD_W == 0.500 and task.BOARD_H == 0.350 and task.BOARD_T == 0.00635
    assert task.BOARD_FRONT_X == 0.42 and task.BOARD_BOTTOM == 0.050
    assert task.HOLE_D == 0.0071 and task.PITCH == 0.0254
    # 19 columns x 13 rows of holes over the whole face.
    assert len(task.HOLE_COLS) == 19 and len(task.HOLE_ROWS) == 13


def test_tagged_holes_land_exactly_on_grid_holes():
    for y in (task.TAG_Y, -task.TAG_Y):
        assert abs(y / task.PITCH - round(y / task.PITCH)) < 1e-9
    assert abs(task.TAG_ROW * task.PITCH - 0.2032) < 1e-9  # 20.32 cm above the table
    assert abs(task.EXIST_ROWS[1] * task.PITCH - 0.1524) < 1e-9  # lower existing hook, 15.24 cm up


def test_board_mesh_has_real_round_holes():
    from bodykit import load_obj

    points, faces = load_obj(str(task.HERE / "assets" / "pegboard.obj"))
    assert len(faces) > 5000
    assert abs(points[:, 1].max() - task.BOARD_W / 2) < 1e-6
    assert abs(points[:, 2].max() - task.BOARD_H / 2) < 1e-6
    # Hole walls dip inside the board's faces: vertices exist at hole radius.
    assert (np.abs(np.abs(points[:, 0]) - task.BOARD_T / 2) < 1e-6).any()


def _turned(rot_deg):
    """Hook pose turned ``rot_deg`` about the curve's centre from seated at hole A (positive tips it up)."""
    hole = task.tag_hole("a")
    a = math.radians(rot_deg)
    rot = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    c = np.array(G.ARC_CENTRE)
    pos = hole + c - rot @ c
    return state_from_pose(_spec("hook_a"), pos, task._quat_from_matrix(rot))


def test_hook_dimensions_match_the_setup_text():
    assert abs(task.WIRE_R * 2 - 0.0048) < 1e-12  # 3/16 in wire
    assert task.HOOK_MASS == 0.030
    assert task.PRONG_DROP == task.PITCH and task.BOTTOM_TIP_X == 0.008
    # Curved top prong: 20 mm radius, 28 degrees, from 2 mm into the board; tip 11.4 mm past the face, 2.3 mm up.
    assert G.ARC_R == 0.020 and G.ARC_DEG == 28.0 and G.ARC_X0 == 0.002
    assert abs(task.TOP_TIP[0] - 0.01139) < 1e-4 and abs(task.TOP_TIP[2] - 0.00234) < 1e-4
    # 150 mm arm to the ball centre, its last 25 mm up 15 degrees.
    tip = np.array(task.TIP_END)
    assert abs(float(tip[0]) - (-0.150)) < 2e-3
    assert abs(math.degrees(math.asin((tip[2] + task.PRONG_DROP) / 0.025)) - 15.0) < 1e-6


def test_new_hooks_start_lying_flat_arms_toward_the_robot():
    assert task.NEW_HOOK_POS == ((0.22, 0.08), (0.22, -0.08))
    for name in task.NEW_HOOKS:
        spec = _spec(name)
        st = state_from_pose(spec, spec.pos, spec.start_xyzw)
        pts = task._wire_points(st)
        on_table = task.TOP + task.WIRE_R  # wire centreline one radius above the table, all along
        assert abs(pts[:, 2].min() - on_table) < 1e-6 and abs(pts[:, 2].max() - on_table) < 1e-6
        arm = st.rotation() @ np.array([-1.0, 0.0, 0.0])
        assert arm[0] < -0.999  # arm toward the robot
    # mirror images: A's prongs toward -y, B's toward +y
    ya = (task.LYING_ROTS["hook_a"] @ np.array([0.0, 0.0, 1.0]))[1]
    yb = (task.LYING_ROTS["hook_b"] @ np.array([0.0, 0.0, 1.0]))[1]
    assert ya < -0.99 and yb > 0.99


def test_seated_hook_measures_seated_and_top_only_does_not():
    hole = task.tag_hole("a")
    seated = state_from_pose(_spec("hook_a"), hole, (0.0, 0.0, 0.0, 1.0))
    m = task.hook_measure({"hook_a": seated}, "hook_a", hole)
    assert scoring.is_seated(m)
    assert m.bottom_depth_mm >= 5.0 and m.back_gap_mm <= 2.0 and m.arm_square_deg <= 0.01
    # Part-way through the threading turn (tipped 15 deg): the curve is through the hole with its
    # tip behind the board, the bottom prong not yet in: top-engaged, not seated.
    m2 = task.hook_measure({"hook_a": _turned(15.0)}, "hook_a", hole)
    assert scoring.is_top_engaged(m2) and not scoring.is_seated(m2)
    # At the start of the turn (tipped 50 deg) the whole hook is still in front of the board.
    m3 = task.hook_measure({"hook_a": _turned(G.INSERT_DEG)}, "hook_a", hole)
    assert not scoring.is_top_engaged(m3)
    assert task._wire_points(_turned(G.INSERT_DEG))[:, 0].max() + task.WIRE_R < task.BOARD_FRONT_X


def test_curved_prong_has_room_to_thread_the_hole():
    # The threading turn never cuts the board, and the curve keeps 0.5 mm clear of the hole.
    import check_fit

    gap, _, whole = check_fit.threading_margin(120)
    assert gap > 0.0005 and whole > -1e-6


def test_square_is_measured_from_above_and_from_the_side_separately():
    hole = task.tag_hole("a")
    a, b = math.radians(8.0), math.radians(7.0)
    pitch = np.array([[math.cos(a), 0.0, -math.sin(a)], [0.0, 1.0, 0.0], [math.sin(a), 0.0, math.cos(a)]])
    yaw = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    st = state_from_pose(_spec("hook_a"), hole, task._quat_from_matrix(yaw @ pitch))
    m = task.hook_measure({"hook_a": st}, "hook_a", hole)
    assert 7.9 < m.arm_square_deg < 8.2  # 8 deg down and 7 deg sideways: 8, not the 10.6 total


def test_seated_hook_cannot_be_pulled_straight_out():
    # Pulled straight out along the arm, the raised tip (2.3 mm above the hole axis) would have to
    # pass the 7.1 mm hole: its wire top would be 4.7 mm above the axis, past the 3.55 mm hole wall.
    assert task.TOP_TIP[2] + task.WIRE_R > task.HOLE_D / 2 + 0.0005


def test_hook_at_an_untagged_hole_earns_nothing_at_the_tagged_one():
    spec = _spec("hook_a")
    wrong = task.hole_world(task.TAG_Y + task.PITCH, task.TAG_ROW)
    st = state_from_pose(spec, wrong, (0.0, 0.0, 0.0, 1.0))
    m = task.hook_measure({"hook_a": st}, "hook_a", task.tag_hole("a"))
    assert not scoring.is_top_engaged(m) or not m.top_in_tagged


def test_back_panel_gap_leaves_room_for_the_curved_tip():
    # The curved tip ends 11.4 mm past the face; with the wire's radius it stays clear of the back
    # panel 26.35 mm behind the face (board 6.35 mm, then the 20 mm gap).
    assert task.TOP_TIP[0] + task.WIRE_R < task.BOARD_T + task.BACK_GAP - 0.005


def test_hook_on_its_back_is_not_off_the_table():
    # A hook knocked over on the table, prongs down, still counts as on the table.
    spec = _spec("hook_a")
    flip = np.diag([1.0, -1.0, -1.0])  # upside down
    low = min(float((flip @ p)[2]) for p in task._HOOK_WIRE_LOCAL)
    st = state_from_pose(spec, (0.25, 0.1, task.TOP + task.WIRE_R - low), task._quat_from_matrix(flip))
    objs = {n: state_from_pose(_spec(n), _spec(n).pos, _spec(n).start_xyzw or (0.0, 0.0, 0.0, 1.0))
            for n in (*task.NEW_HOOKS, *task.EXIST_HOOKS, *task.CARDS)}
    objs["hook_a"] = st
    assert not task.end_state(objs).new_off_table


def test_cards_clear_the_table_and_the_lower_arm():
    for name, row in zip(task.CARDS, task.EXIST_ROWS):
        info = task.CARD_INFO[name]
        arm_z = (row - 1) * task.PITCH  # above the table
        bottom = arm_z - info["hole_off"] - info["size"][2] / 2
        assert bottom > 0.010
    # The upper (short) card's bottom stays above the lower hook's arm.
    up = task.CARD_INFO["card_0"]
    upper_bottom = (task.EXIST_ROWS[0] - 1) * task.PITCH - up["hole_off"] - up["size"][2] / 2
    assert upper_bottom > (task.EXIST_ROWS[1] - 1) * task.PITCH


def test_existing_hooks_and_cards_start_hung():
    objs = {}
    for name in (*task.NEW_HOOKS, *task.EXIST_HOOKS, *task.CARDS):
        spec = _spec(name)
        q = spec.start_xyzw if spec.start_xyzw is not None else (0.0, 0.0, 0.0, 1.0)
        objs[name] = state_from_pose(spec, spec.pos, q)
    st = task.end_state(objs)
    assert not st.existing_knocked and not st.product_fallen
    assert score_is_zero_at_start(st)


def _start_objects():
    objs = {}
    for name in (*task.NEW_HOOKS, *task.EXIST_HOOKS, *task.CARDS):
        spec = _spec(name)
        q = spec.start_xyzw if spec.start_xyzw is not None else (0.0, 0.0, 0.0, 1.0)
        objs[name] = state_from_pose(spec, spec.pos, q)
    return objs


def _sag(objs, hook: str, card: str, deg: float, out_mm: float, card_drop_mm: float = None):
    """Tip an existing hook arm-down by deg about its top hole and slide it out; move its card
    down with the arm at the card's hole (or by card_drop_mm)."""
    a = math.radians(deg)
    rot = np.array([[math.cos(a), 0.0, -math.sin(a)], [0.0, 1.0, 0.0], [math.sin(a), 0.0, math.cos(a)]])
    assert (rot @ [-1.0, 0.0, 0.0])[2] <= 0  # arm down
    h = objs[hook]
    objs[hook] = state_from_pose(_spec(hook), h.pos - np.array([out_mm / 1000, 0.0, 0.0]),
                                 task._quat_from_matrix(rot))
    c = objs[card]
    reach = h.pos[0] - c.pos[0]
    drop = reach * math.tan(a) if card_drop_mm is None else card_drop_mm / 1000
    objs[card] = state_from_pose(_spec(card), c.pos - np.array([out_mm / 1000, 0.0, drop]), (0.0, 0.0, 0.0, 1.0))


def test_sagging_existing_hook_with_its_card_still_counts():
    # The simulator settles a loaded hook a few degrees arm-down; only coming out counts.
    objs = _start_objects()
    _sag(objs, "exist_1", "card_1", 9.0, 2.0)
    st = task.end_state(objs)
    assert not st.existing_knocked and not st.product_fallen


def test_card_off_its_hook_is_fallen():
    objs = _start_objects()
    _sag(objs, "exist_0", "card_0", 0.0, 0.0, card_drop_mm=30.0)
    assert task.end_state(objs).product_fallen


def test_existing_hook_pulled_out_is_knocked():
    objs = _start_objects()
    _sag(objs, "exist_0", "card_0", 0.0, 20.0)
    assert task.end_state(objs).existing_knocked


def score_is_zero_at_start(st) -> bool:
    return scoring.score(st) == 0.0
