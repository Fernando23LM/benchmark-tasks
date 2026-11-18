"""Peg hook geometry, shared by gen_assets.py (the mesh), task.py (scoring) and demo.py.

Hook frame (metres): origin at the top hole's centre on the board's front face, +x into the
board, +z up; a seated hook has this frame aligned with the world. Pure Python.

The hook is 3/16 in (4.8 mm) zinc-plated steel wire:
- a back, vertical, one wire radius in front of the board face;
- a top prong: straight from the back to 2 mm into the board, then curving *up* on a 20 mm
  radius for 28 degrees (tip 11.4 mm behind the face and 2.3 mm above the hole's axis). A curve follows itself, so the prong threads the hole by
  turning the hook about the curve's centre; a sharp 90 degree bend cannot pass a 7.1 mm hole
  in a 6.35 mm board at all. Pulled straight out, the raised tip catches on the board;
- a straight bottom prong 8 mm past the face, one pitch (25.4 mm) below the top prong;
- a 150 mm arm whose last 25 mm angle up 15 degrees, with a ball end.

Installing: hold the hook tipped up by INSERT_DEG about ARC_CENTRE (arm raised toward the
robot, the whole hook in front of the board), then turn it level about ARC_CENTRE. The curved
prong threads the tagged hole and, at the end of the same turn, the bottom prong swings
straight into the hole below.
"""

from __future__ import annotations

import math

BOARD_T = 0.00635
PITCH = 0.0254

WIRE_D = 0.0048  # 3/16 in
WIRE_R = WIRE_D / 2
HOOK_MASS = 0.030  # 4.8 mm steel wire, about 230 mm long, plus the ball end

BACK_X = -WIRE_R  # back wire centre, touching the board face
PRONG_DROP = PITCH  # bottom prong one pitch below the top prong
BOTTOM_TIP_X = 0.008  # bottom prong tip, past the front face

ARC_X0 = 0.002  # top prong turns into its curve 2 mm into the board (check_fit.py: 0.58 mm threading room, 5.5 deg worst sag)
ARC_R = 0.020
ARC_DEG = 28.0
ARC_CENTRE = (ARC_X0, 0.0, ARC_R)  # the hook turns about this point to thread the prong
TOP_TIP = (ARC_X0 + ARC_R * math.sin(math.radians(ARC_DEG)), 0.0,
           ARC_R * (1.0 - math.cos(math.radians(ARC_DEG))))

ARM_END_X = -0.125  # straight arm ends here; the upturned 25 mm tip follows
TIP_UP_DEG = 15.0
TIP_END = (ARM_END_X - 0.025 * math.cos(math.radians(TIP_UP_DEG)), 0.0,
           -PRONG_DROP + 0.025 * math.sin(math.radians(TIP_UP_DEG)))
BALL_R = 0.003  # 6 mm ball end

INSERT_DEG = 50.0  # tipped-up start of the threading turn: the whole hook is in front of the face


def arc_points(n: int = 8) -> list[tuple[float, float, float]]:
    """Centreline of the top prong's curve, from the straight part to the tip (n+1 points)."""
    out = []
    for k in range(n + 1):
        a = math.radians(ARC_DEG) * k / n
        out.append((ARC_X0 + ARC_R * math.sin(a), 0.0, ARC_R * (1.0 - math.cos(a))))
    return out


def top_prong_points() -> list[tuple[float, float, float]]:
    """Top prong centreline: from the back's top corner, straight to mid-board, then the curve."""
    return [(BACK_X, 0.0, 0.0), *arc_points()]


def wire_polyline() -> list[tuple[float, float, float]]:
    """The whole hook's centreline as one path, for gripper-clearance checks."""
    return [*reversed(top_prong_points()), (BACK_X, 0.0, -PRONG_DROP), (BOTTOM_TIP_X, 0.0, -PRONG_DROP),
            (BACK_X, 0.0, -PRONG_DROP), (ARM_END_X, 0.0, -PRONG_DROP), TIP_END]


def segments() -> list[tuple[tuple, tuple]]:
    """Straight wire pieces of the mesh (the curve as a chain of short pieces)."""
    top = top_prong_points()
    segs = list(zip(top[:-1], top[1:]))
    segs += [((BACK_X, 0.0, 0.0), (BACK_X, 0.0, -PRONG_DROP)),
             ((BACK_X, 0.0, -PRONG_DROP), (BOTTOM_TIP_X, 0.0, -PRONG_DROP)),
             ((BACK_X, 0.0, -PRONG_DROP), (ARM_END_X, 0.0, -PRONG_DROP)),
             ((ARM_END_X, 0.0, -PRONG_DROP), TIP_END)]
    return segs
