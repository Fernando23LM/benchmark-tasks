"""Simulator probe: the existing hooks and cards from reset through the first 5 s, arms still.

Prints, at reset and then every few steps, each existing hook's position and how far its arm
points off square, every field of the scorer's seated test at its own holes, and each card's
hang-hole distance from its arm and tilt. Also the spawn poses, to tell a hook that moves from one
the scorer misreads.

    python probe_start.py task.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def main(task_path: str, emb=None) -> None:
    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    import demo
    import scoring

    if emb is None:
        emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=20, cam_width=320, cam_height=176)
    emb.reset(Scene(id="probe", instruction="probe"))
    T = emb.task.module
    ctx = demo.Ctx(emb, T)
    for spec in T.OBJECTS:
        if spec.name in (*T.EXIST_HOOKS, *T.CARDS):
            print(f"spawn {spec.name} pos={np.round(spec.pos, 4).tolist()} quat={spec.start_xyzw}", flush=True)

    def report(step: int) -> None:
        st = ctx.states()
        for name, row in zip(T.EXIST_HOOKS, T.EXIST_ROWS):
            s = st[name]
            m = T.hook_measure(st, name, T.hole_world(0.0, row))
            arm = s.rotation() @ np.array([-1.0, 0.0, 0.0])
            print(f"step {step:3d} {name} pos={np.round(s.pos, 4).tolist()} arm_dir={np.round(arm, 3).tolist()} "
                  f"top_in={m.top_in_tagged} tip_behind={m.top_tip_behind} bottom={m.bottom_depth_mm:.1f}mm "
                  f"back={m.back_gap_mm:.1f}mm square={m.arm_square_deg:.1f}deg seated={scoring.is_seated(m)}",
                  flush=True)
        for name, row, hook in zip(T.CARDS, T.EXIST_ROWS, T.EXIST_HOOKS):
            s = st[name]
            hole = s.pos + s.rotation() @ np.array([0.0, 0.0, T.CARD_INFO[name]["hole_off"]])
            arm_z = T.TOP + (row - 1) * T.PITCH
            a = np.array([T.BOARD_FRONT_X - 0.005, 0.0, arm_z])
            b = np.array([T.BOARD_FRONT_X - 0.150, 0.0, arm_z])
            d = T._seg_point_dist(hole, a, b)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(s.rotation()[2, 2])))))
            print(f"step {step:3d} {name} pos={np.round(s.pos, 4).tolist()} hole={np.round(hole, 4).tolist()} "
                  f"arm_z={arm_z:.4f} hole_to_arm={d * 1000:.1f}mm tilt={tilt:.1f}deg hung={T._product_hung(st, name, hook)} "
                  f"to_wire={T._dist_to_polyline(hole, T._wire_points(st[hook])) * 1000:.1f}mm",
                  flush=True)

    report(0)
    for step in range(1, 201):
        emb.step(ctx.action())
        if step in (1, 5, 20, 60, 100, 200):
            report(step)
    print("probe done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
    import os

    os._exit(0)
