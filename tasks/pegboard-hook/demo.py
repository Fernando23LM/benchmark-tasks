"""Scripted full-score demo for the pegboard-hook task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record] [--cam-width 1280]
    python demo.py task.py - --dry          # check the plan's IK without starting the simulator

The left arm installs hook A, then the right arm hook B. Each arm picks its hook up by
the arm wire, turns it to face the board tipped up 50 degrees just in front of its
tagged hole, and turns it level about the centre of the top prong's curve: the curve
threads the hole and curls in behind the board, and at the end of the same turn the
bottom prong swings into the hole below. The arm presses the back flush, lets go,
backs off the wire and goes home. Each gripper works from the outside of its hook,
76 mm from the centre hooks and their cards.

The plan comes from the task's fixed layout (the run uses
object_jitter=0). The demo looks at where each hook actually is before
each grasp and while it seats (the hooks are in plain view of the
cameras); it never moves objects or edits the simulation. DEMO_VARIANT
selects wrong strategies used to test the scorer (see the README).
Writes one MP4 per camera, setup.png (the first top-camera frame),
final.png and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


class Arm:
    """One arm: Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.hold = (np.zeros(3), np.eye(3))
        self.bias = (np.eye(3), np.zeros(3))  # world (rotation, offset) applied to held-hook commands
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.min_margin = math.inf
        self.phase = "start"
        self.by_phase: dict[str, list[float]] = {}

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=80)
        self._record(pe, re, q)
        self.last_pe, self.last_target = pe, (np.asarray(pos, float).copy(), np.asarray(rot, float).copy())
        if pe > 0.03:  # unreachable target: keep the last good command
            return
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, np.asarray(rot, float), s))
            yield

    def _record(self, pe: float, re: float, q) -> None:
        from yam_arena import ik

        margin = float(np.min(np.minimum(q - ik.LOWER, ik.UPPER - q)))
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re))
        self.min_margin = min(self.min_margin, margin)
        rec = self.by_phase.setdefault(self.phase, [0.0, 0.0, math.inf, 0.0])
        rec[:] = [max(rec[0], pe * 1000), max(rec[1], math.degrees(re)), min(rec[2], margin), float(q[5])]

    def goto(self, pos, rot, duration: float, j6: float | None = None, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` by interpolating the joints, for big wrist turns."""
        from yam_arena import ik

        seeds = [self.q.copy()]
        rolls = (j6, j6 + 0.4, j6 - 0.4) if j6 is not None else (0.0, 1.0, -1.0, 2.0, -2.0)
        for roll in rolls:
            seed = self.q.copy()
            seed[5] = roll
            seeds.append(seed)
        best, best_cost, fallback = None, math.inf, None
        for seed in seeds:
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(np.asarray(rot, float)), seed,
                                 base_pos=self.base, iters=200)
            if fallback is None or pe + 0.05 * re < fallback[1] + 0.05 * fallback[2]:
                fallback = (q, pe, re)
            if pe < 1e-3 and re < 0.02:
                cost = abs(q[5] - j6) if j6 is not None else float(np.abs(q - self.q).sum())
                if cost < best_cost:
                    best, best_cost = (q, pe, re), cost
        target, pe, re = best if best is not None else fallback
        if best is None and pe > 0.03:
            target = self.q.copy()
        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q, self.grip = q0 + s * (target - q0), g0 + s * (g1 - g0)
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
            self.cmd = np.r_[self.q, self.grip]
            self._record(pe, re, self.q)
            yield

    def go_home(self, duration: float = 2.5):
        yield from self.goto(*self.home, duration, j6=0.0, grip=1.0)

    def calibrate(self, g_pos, g_rot) -> None:
        g_rot = np.asarray(g_rot, float)
        self.hold = (g_rot.T @ (self.pos - np.asarray(g_pos, float)), g_rot.T @ self.rot)

    def calibrate_actual(self, ctx, name: str, planned) -> None:
        g_pos, g_rot = ctx.look(name, planned)
        if ctx.emb is None:
            self.calibrate(g_pos, g_rot)
            return
        tcp_pos, tcp_rot = ctx.tcp(self.name)
        g_rot = np.asarray(g_rot, float)
        self.hold = (g_rot.T @ (tcp_pos - np.asarray(g_pos, float)), g_rot.T @ tcp_rot)

    def holding(self, g_pos, g_rot):
        """TCP command that puts the held hook at ``g_pos``/``g_rot``, with the arm's measured
        settling error (``bias``, a world transform set by aim()) taken out."""
        g_rot = np.asarray(g_rot, float)
        pos, rot = np.asarray(g_pos, float) + g_rot @ self.hold[0], g_rot @ self.hold[1]
        b_rot, b_pos = self.bias
        return b_rot @ pos + b_pos, b_rot @ rot

    def correct(self, seen_pos, seen_rot, target_pos, target_rot, max_mm: float = 25.0,
                gain: float = 1.0) -> float:
        """Fold ``gain`` of the hook's remaining error into ``bias``: the world move turning the
        hook about its origin toward the target rotation and shifting it toward the target
        position, applied to every later command. Returns the error in mm."""
        seen_pos, target_pos = np.asarray(seen_pos, float), np.asarray(target_pos, float)
        d_rot = _scale_rot(np.asarray(target_rot, float) @ np.asarray(seen_rot, float).T, gain)
        err = float(np.linalg.norm(target_pos - seen_pos)) * 1000
        if err > max_mm:  # not a settling error: leave it
            return err
        d_pos = seen_pos - d_rot @ seen_pos + gain * (target_pos - seen_pos)
        b_rot, b_pos = self.bias
        self.bias = (d_rot @ b_rot, d_rot @ b_pos + d_pos)
        return err

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def follow(self, pose_at, duration: float):
        """Generator: track ``pose_at(s)`` -> (hook pos, hook rot) for s from 0 to 1."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self._solve(*self.holding(*pose_at(ik.smoothstep(k / n))))
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _scale_rot(r: np.ndarray, g: float) -> np.ndarray:
    """The rotation ``g`` of the way along ``r`` (same axis, g times the angle)."""
    ang = math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))
    if ang < 1e-9:
        return np.eye(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    axis = axis / (np.linalg.norm(axis) or 1.0)
    a = g * ang
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * k @ k


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def down_grip(yaw: float = 0.0) -> np.ndarray:
    """TCP rotation of a straight-down grip turned ``yaw`` about the vertical."""
    from yam_arena.demo import grip_rotation

    return grip_rotation(yaw)


def aim(arm: Arm, ctx, name: str, target_pos, target_rot, duration: float = 2.0, rounds: int = 3):
    """Generator: re-calibrate the grip from the hook's measured pose and the arm's actual TCP,
    move the hook to the target pose, then measure where it really is and correct the command by
    what is left, a few times (the arm settles some mm off its command; the curved prong has to
    reach its hole within about 1 mm)."""
    yield from wait(0.3)
    arm.calibrate_actual(ctx, name, (target_pos, target_rot))
    yield from arm.move(*arm.holding(target_pos, target_rot), duration)
    yield from wait(0.3)
    if ctx.emb is None:
        return
    for _ in range(rounds):
        seen_pos, seen_rot = ctx.look(name, (target_pos, target_rot))
        if arm.correct(seen_pos, seen_rot, target_pos, target_rot) < 0.3:
            return
        yield from arm.move(*arm.holding(target_pos, target_rot), 0.6)
        yield from wait(0.4)


def turned_pose(task, hole, deg: float, back: float = 0.0):
    """Hook pose (origin, rotation) turned ``deg`` about its curve's centre from seated at ``hole``
    (positive tips the arm up), moved ``back`` metres straight out from the board."""
    import hookgeom as G

    rot = _ry(math.radians(deg))
    c = np.array(G.ARC_CENTRE)
    return hole + c - rot @ c + np.array([-back, 0.0, 0.0]), rot


OVERTURN_DEG = float(os.environ.get("DEMO_OVERTURN_DEG", "1.5"))  # turn slightly past level: bottom prong pushed home
TURN_S = float(os.environ.get("DEMO_TURN_S", "6.0"))  # the threading turn, from INSERT_DEG to level
SERVO_GAIN = float(os.environ.get("DEMO_SERVO", "0.15"))  # share of the hook's error steered out per step


def hook_work(task, ctx, arm: Arm, name: str, variant: str):
    """Install one new hook: grasp its arm, tip it up in front of its tagged hole, turn it level
    about the curve's centre (the curved prong threads the hole and the bottom prong swings into
    the hole below), press it flush, let go and back away."""
    import hookgeom as G

    T = task
    # DEMO_AIR_MM (a test): do everything that far in front of the board, so the turn touches nothing
    hole = T.tag_hole(name[-1]) - np.array([float(os.environ.get("DEMO_AIR_MM", "0")) / 1000.0, 0.0, 0.0])
    lying = T.LYING_ROTS[name]
    start = np.array([*T.NEW_HOOK_POS[0 if name == "hook_a" else 1], T.TOP + T.WIRE_R])
    # Grip: fingers close across the wire (along the hook's back), wrist rolled DEMO_GRIP_YAW and
    # the gripper tilted DEMO_GRIP_TILT about the world y axis, so it comes in from the robot's
    # side at 45 degrees. The hook then cannot pitch in the fingers (pitching would turn the wire
    # out of the pads' plane). check_reach.py chose this grip and the grasp point DEMO_GRASP_X
    # (100 mm out along the arm): through the whole sequence every joint keeps 0.33 rad to its
    # stops, where a straight-down or mid-arm grip runs the wrist (joint 5 or 6) into its limit.
    yaw = float(os.environ.get("DEMO_GRIP_YAW", str(math.pi)))
    tilt = math.radians(float(os.environ.get("DEMO_GRIP_TILT", "-45")))
    grip = _ry(tilt) @ down_grip(yaw)
    approach = grip[:, 2]  # TCP z, from the wrist toward the fingertips

    arm.phase = f"{name} grasp"
    gx = float(os.environ.get("DEMO_GRASP_X", "-0.100"))
    grasp_local = np.array([gx, 0.0, -T.PRONG_DROP])
    # TCP 6 mm back along the approach from the wire's centre (found by trial grasps in the
    # simulator: there the pads bite and lift it; further back they close above it, closer in
    # they dig into the table and drag it). DEMO_GRASP_BACK_MM tunes it.
    above = -approach * float(os.environ.get("DEMO_GRASP_BACK_MM", "6")) / 1000.0
    if ctx.emb is not None:
        st = ctx.states()[name]
        seen_grasp = st.pos + st.rotation() @ grasp_local + above
    else:
        seen_grasp = start + lying @ grasp_local + above
    # Joint-space move to 60 mm back along the approach (a Cartesian move from the ready pose
    # would roll the wrist into its stop on the way).
    yield from arm.goto(seen_grasp - 0.06 * approach, grip, 3.0, grip=1.0)
    for _attempt in range(2):
        yield from arm.move(seen_grasp - 0.06 * approach, grip, 1.0, grip=1.0)
        yield from arm.move(seen_grasp, grip, 2.0)
        yield from wait(0.6)  # let the arm's approach lag settle before closing
        yield from arm.set_grip(0.0, 1.2)
        yield from wait(0.4)
        arm.calibrate_actual(ctx, name, (start, lying))
        yield from arm.move(arm.pos + [0.0, 0.0, 0.018], grip, 1.0)
        if ctx.emb is None:
            break
        seen, srot = ctx.look(name, (start, lying))
        if float(seen[2]) > float(start[2]) + 0.008:
            break  # it came up with us
        yield from arm.move(arm.pos + [0.0, 0.0, -0.018], grip, 1.0)
        yield from arm.set_grip(1.0, 0.8)
        yield from arm.move(arm.pos + [0.0, 0.0, 0.05], grip, 1.2, grip=1.0)
        yield from wait(0.3)
        # Retry once from the hook's measured pose, only if it still lies flat near its start.
        flat = float(np.dot(srot @ np.array([-1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0]))
        near = float(np.linalg.norm(seen[:2] - start[:2])) < 0.05
        if not (near and flat > math.cos(math.radians(25.0))):
            break
        seen_grasp = seen + srot @ grasp_local + above

    arm.phase = f"{name} lift and tip"
    yield from arm.move(arm.pos + [0.0, 0.0, 0.06], grip, 1.2)
    # Turn the hook to face the board, tipped up INSERT_DEG, 30 mm out from the board, then come
    # straight in until the curved tip is just in front of its hole (the whole hook still clear of
    # the board: see check_fit.py).
    far = turned_pose(T, hole, G.INSERT_DEG, back=0.030)
    yield from arm.move(*arm.holding(*far), 3.5)
    yield from aim(arm, ctx, name, *far, duration=1.0)  # re-aim from the hook's measured pose
    near = turned_pose(T, hole, G.INSERT_DEG, back=0.001)
    yield from arm.move(*arm.holding(*near), 1.5)
    yield from aim(arm, ctx, name, *near, duration=0.8)
    where(ctx, name, "before the turn", near, arm)

    arm.phase = f"{name} turn in"
    last = 15.0 if variant == "top_only" else -OVERTURN_DEG
    seconds = TURN_S * (G.INSERT_DEG - last) / G.INSERT_DEG
    yield from turn(arm, ctx, name, T, hole, G.INSERT_DEG, last, seconds)
    where(ctx, name, f"after the turn to {last:.1f} deg", turned_pose(T, hole, last), arm)
    if variant == "top_only":  # let go part-way: curve through the hole, bottom prong not in
        yield from release(arm)
        return
    # Press flush: level, 1.5 mm past the face, so the arm's stiffness holds the back on it.
    yield from arm.move(*arm.holding(*turned_pose(T, hole, 0.0, back=-0.0015)), 0.8)
    yield from wait(0.5)
    where(ctx, name, "pressed", turned_pose(T, hole, 0.0), arm)
    if variant == "hold":  # keep hold of the seated hook: the release points are lost
        return
    arm.phase = f"{name} let go"
    yield from release(arm)
    where(ctx, name, "let go", turned_pose(T, hole, 0.0))


def _rot_diff(a, b) -> tuple[float, np.ndarray]:
    rel = np.asarray(a).T @ np.asarray(b)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) / 2.0))))
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    return ang, axis / (np.linalg.norm(axis) or 1.0)


def where(ctx, name: str, label: str, expected, arm=None) -> None:
    """Print how far the hook is from where the script means it to be (world mm, and its turn off
    plan, about a world axis), and with ``arm``, how far it has slipped in the fingers since the
    grip was last calibrated. Simulator runs only."""
    if ctx.emb is None:
        return
    st = ctx.states()[name]
    d = (st.pos - np.asarray(expected[0])) * 1000.0
    ang, axis = _rot_diff(expected[1], st.rotation())
    axis = np.asarray(expected[1]) @ axis
    msg = (f"WHERE {name} {label}: off by ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) mm, turned {ang:.1f} deg "
           f"about ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})")
    if arm is not None and arm.hold is not None:
        tcp_pos, tcp_rot = ctx.tcp(arm.name)
        r = st.rotation()
        now = (r.T @ (tcp_pos - st.pos), r.T @ tcp_rot)
        sang, _ = _rot_diff(arm.hold[1], now[1])
        msg += f"; slipped {np.linalg.norm(now[0] - arm.hold[0]) * 1000:.1f} mm {sang:.1f} deg in the grip"
        lag = (tcp_pos - arm.pos) * 1000.0
        lang, _ = _rot_diff(arm.rot, tcp_rot)
        msg += f"; arm off its command ({lag[0]:+.1f}, {lag[1]:+.1f}, {lag[2]:+.1f}) mm {lang:.1f} deg"
    print(msg, flush=True)


def turn(arm: Arm, ctx, name: str, T, hole, deg0: float, deg1: float, seconds: float):
    """Generator: turn the held hook about its curve's centre from ``deg0`` to ``deg1``, with the
    grip calibrated once from the measured hook at the start (recalibrating during the turn would
    re-anchor the target to a hook held back by contact, and no pressing force would build)."""
    if ctx.emb is not None:
        yield from wait(0.2)
        arm.calibrate_actual(ctx, name, turned_pose(T, hole, deg0))
    from yam_arena import ik

    total = max(1, round(seconds * CONTROL_HZ))
    for n in range(1, total + 1):
        deg = deg0 + ik.smoothstep(n / total) * (deg1 - deg0)
        target = turned_pose(T, hole, deg)
        if ctx.emb is not None and SERVO_GAIN > 0:
            # The arm lags its command by up to 12 mm while moving (it catches up when it stops):
            # steer by the hook's measured error from the plan, a little every step.
            seen = ctx.look(name, target)
            arm.correct(*seen, *target, gain=SERVO_GAIN)
        arm._solve(*arm.holding(*target))
        if n % 10 == 0:
            where(ctx, name, f"turning, {deg:.1f} deg", target, arm)
        yield
    for _ in range(round(0.5 * CONTROL_HZ)):  # settle on the last pose, still steering
        if ctx.emb is not None and SERVO_GAIN > 0:
            arm.correct(*ctx.look(name, target), *target, gain=SERVO_GAIN)
            arm._solve(*arm.holding(*target))
        yield


def release(arm: Arm):
    """Generator: open the fingers, back straight off the wire along the approach, then go home."""
    arm.bias = (np.eye(3), np.zeros(3))
    yield from arm.set_grip(1.0, 0.8)
    approach = arm.rot[:, 2]  # TCP z: from the wrist toward the fingertips
    yield from arm.move(arm.pos - 0.06 * approach, arm.rot, 1.5)
    yield from arm.move(arm.pos + np.array([-0.06, 0.0, 0.03]), arm.rot, 1.2)
    yield from arm.go_home()


def script(task, ctx, variant: str = ""):
    """The demo as a generator: one yield per control step. The left arm installs hook A and the
    right arm hook B, one after the other, each from its own side, so neither gripper comes near
    the products hanging in the centre column."""
    if variant == "none":
        yield from wait(5.0)
        return
    ctx.flags.clear()
    yield from hook_work(task, ctx, ctx.left, "hook_a", "" if variant == "hold" else variant)
    if variant == "one_hook":  # only hook A: 0.4
        yield from wait(1.5)
        return
    yield from hook_work(task, ctx, ctx.right, "hook_b", variant)
    yield from wait(1.5)


class Ctx:
    def __init__(self, emb=None, task=None):
        from yam_arena import station

        self.emb, self.task = emb, task
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.flags: dict[str, bool] = {}

    def tcp(self, side: str):
        from bodykit import as_numpy, rotation
        from yam_arena.constants import TCP_OFFSET

        base = self.emb._env.unwrapped
        origin = as_numpy(base.scene.env_origins[0])
        robot = base.scene[f"{side}_robot"]
        i = next(j for j, n in enumerate(robot.body_names) if n == "gripper" or n.endswith("/gripper"))
        g = as_numpy(robot.data.body_link_pose_w)[0][i]
        rot = rotation(g[3:7])
        return g[:3] - origin + rot @ np.array(TCP_OFFSET), rot

    def look(self, name: str, planned):
        if self.emb is None:
            return np.asarray(planned[0], float), np.asarray(planned[1], float)
        state = self.states()[name]
        return state.pos, state.rotation()

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def describe(task, states) -> str:
    import scoring

    st = task.end_state(states)
    parts = []
    for name, m in zip(task.NEW_HOOKS, st.hooks):
        kind = "seated" if scoring.is_seated(m) else ("top" if scoring.is_top_engaged(m) else "no")
        parts.append(f"{name}={kind}(bot{m.bottom_depth_mm:.0f}mm back{m.back_gap_mm:.1f}mm sq{m.arm_square_deg:.0f}d)")
    return (f"{' '.join(parts)} clear={st.clear_s:.2f}s product_fallen={st.product_fallen} "
            f"existing_knocked={st.existing_knocked} off_table={st.new_off_table}")


def dry_run(task_path: str, variant: str) -> dict:
    from yam_arena import task_file

    task = task_file.load(task_path).module
    ctx = Ctx(None, task)
    steps = sum(1 for _ in script(task, ctx, variant))
    result = {"steps": steps}
    for arm in (ctx.left, ctx.right):
        result[arm.name] = {"worst_ik_mm": round(arm.worst_mm, 2), "worst_ik_deg": round(arm.worst_deg, 2),
                            "min_joint_margin_rad": round(arm.min_margin, 3)}
        for phase, (mm, deg, margin, j6) in arm.by_phase.items():
            print(f"{arm.name:5s} {phase:24s} worst {mm:7.2f} mm {deg:6.2f} deg  joint margin {margin:5.2f} rad"
                  f"  ends with joint6 {j6:+.2f}")
    print(json.dumps(result))
    return result


def _render_at(width: int, height: int) -> None:
    import importlib.abc
    import importlib.util

    class Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        NAME = "yam_arena.embodiment"

        def find_spec(self, name, path, target=None):
            if name != self.NAME:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            self._inner = spec.loader
            spec.loader = self
            return spec

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            cls = module.BimanualYamEmbodiment
            init = cls.__init__

            def patched(obj, *args, **kwargs):
                kwargs.setdefault("cam_height", height)
                kwargs.setdefault("cam_width", width)
                init(obj, *args, **kwargs)

            cls.__init__ = patched

    sys.meta_path.insert(0, Hook())


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 2.0, variant: str = "", log_every: int = 100, emb=None) -> dict:
    """Play the script in the simulator; ``emb`` is a ready embodiment (headless.py's, without cameras)."""
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if emb is None:
        if cam_width > 640:
            _render_at(cam_width, cam_height)
        emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                                   cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb, task)
    print("start\n ", describe(task, ctx.states()), flush=True)
    gen = script(task, ctx, variant)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % log_every == 0:
            print(f"step {steps} score {info.get('score')} L[{ctx.left.phase}] R[{ctx.right.phase}] "
                  f"{describe(task, ctx.states())}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    print(f"end at step {steps}\n ", describe(task, ctx.states()), flush=True)
    for arm in (ctx.left, ctx.right):
        for phase, (mm, deg, margin, j6) in arm.by_phase.items():
            print(f"{arm.name:5s} {phase:24s} worst {mm:7.2f} mm {deg:6.2f} deg margin {margin:5.2f}", flush=True)
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant or "demo",
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
            imageio.imwrite(out / "final.png", frames[top][-1])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--cam-width", type=int, default=320)
    a = ap.parse_args()
    v = os.environ.get("DEMO_VARIANT", "")
    if a.dry:
        dry_run(a.task, v)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record, variant=v, log_every=a.log_every,
            cam_width=a.cam_width, cam_height=a.cam_width * 9 // 16 if a.cam_width > 320 else 176)
        os._exit(0)
