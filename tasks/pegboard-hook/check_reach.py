"""Offline reach check for the demo's grasp (no simulator). For each grip (wrist roll ``yaw`` and
tilt about the world y axis ``tilt`` at the grasp) and grasp point on the arm wire, it follows the
demo's sequence with the arm's IK, each solve seeded from the last as the demo does: above the
hook, at the grasp, lifted, tipped up INSERT_DEG 30 mm out, then every 2.5 degrees of the turn
to level. It prints the worst position and orientation error and the smallest joint margin with
the joint and pose where it occurs. The best grips come last.

    python check_reach.py                    # hook A, left arm (hook B on the right arm is its mirror)
"""

from __future__ import annotations

import math

import numpy as np

import hookgeom as G


def grip_rotation(yaw: float, tilt: float) -> np.ndarray:
    import demo

    return demo._ry(tilt) @ demo.down_grip(yaw)


def follow(yaw: float, tilt: float, gx: float, side: str = "left"):
    from yam_arena import ik, station
    from yam_arena.constants import READY

    import demo
    import task as T

    name = "hook_a" if side == "left" else "hook_b"
    base = np.array(station.BASES[side], float)
    hole = T.tag_hole(name[-1])
    lying = T.LYING_ROTS[name]
    start = np.array([*T.NEW_HOOK_POS[0 if side == "left" else 1], T.TOP + T.WIRE_R])
    grip = grip_rotation(yaw, tilt)
    tcp = start + lying @ np.array([gx, 0.0, -T.PRONG_DROP]) + np.array([0.0, 0.0, T.WIRE_R + 0.003])
    hold_p, hold_r = lying.T @ (tcp - start), lying.T @ grip

    targets = [("above", tcp + [0, 0, 0.06], grip), ("grasp", tcp, grip), ("lifted", tcp + [0, 0, 0.08], grip)]
    for k, s in enumerate(np.linspace(0, 1, 7)[1:]):  # carry: slerp to the tipped far pose
        p, r = demo.turned_pose(T, hole, G.INSERT_DEG, back=0.03)
        far_p, far_r = p + r @ hold_p, r @ hold_r
        targets.append((f"carry {k}", (1 - s) * (tcp + [0, 0, 0.08]) + s * far_p, ik.slerp_matrix(grip, far_r, s)))
    for deg in np.arange(G.INSERT_DEG, -demo.OVERTURN_DEG - 0.01, -2.5):
        p, r = demo.turned_pose(T, hole, float(deg))
        targets.append((f"turn {deg:.1f}", p + r @ hold_p, r @ hold_r))

    # first solve as Arm.goto does: several wrist-roll seeds, keep the cleanest
    best = None
    for j6 in (0.0, 1.0, -1.0, 2.0, -2.0):
        seed = np.array(READY, float)
        seed[5] = j6
        q, pe, re = ik.solve(targets[0][1], ik.matrix_to_quat_xyzw(targets[0][2]), seed, base_pos=base, iters=300)
        cost = pe + 0.05 * re
        if best is None or cost < best[0]:
            best = (cost, q)
    q = best[1]
    worst_mm = worst_deg = 0.0
    margin, why = math.inf, ""
    for label, pos, rot in targets:
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q, base_pos=base, iters=120)
        m = np.minimum(q - ik.LOWER, ik.UPPER - q)
        j = int(np.argmin(m))
        if m[j] < margin:
            margin, why = float(m[j]), f"joint {j + 1} at {label}"
        worst_mm, worst_deg = max(worst_mm, pe * 1e3), max(worst_deg, math.degrees(re))
    return worst_mm, worst_deg, margin, why


def main():
    rows = []
    for yaw in (0.0, math.pi):
        for tilt_deg in range(-60, 61, 15):
            for gx in (-0.06, -0.08, -0.10, -0.11):
                mm, deg, margin, why = follow(yaw, math.radians(tilt_deg), gx)
                rows.append((margin if mm < 1 and deg < 1 else -1, yaw, tilt_deg, gx, mm, deg, margin, why))
    rows.sort()
    print(f"{'yaw':>5} {'tilt':>5} {'grasp_x':>8} {'worst mm':>9} {'worst deg':>9} {'margin':>7}  limiting joint")
    for _, yaw, tilt_deg, gx, mm, deg, margin, why in rows[-15:]:
        print(f"{yaw:5.2f} {tilt_deg:5d} {gx:8.3f} {mm:9.2f} {deg:9.2f} {margin:7.2f}  {why}")


if __name__ == "__main__":
    main()
