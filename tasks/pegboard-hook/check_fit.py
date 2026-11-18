"""Offline fit check of the hook in the board (no simulator), on the 2D section through the hole
axis, with the board as solid material around round holes and the wire as a tube of its real
radius. Two numbers:

- threading margin: the smallest gap between the wire and the board during the threading turn
  (INSERT_DEG down to level about ARC_CENTRE). It must stay positive, and the bigger it is, the
  more slop the robot's turn can have.
- droop: how far a seated hook can tip arm-down when it is free to slide out and drop in its
  holes. Every pose the wire can reach from the seated pose without entering the board counts
  (a flood fill over sideways shift, height and tip angle), so this is the worst case, before
  friction (the simulator settles about 1.5 degrees past it). The scorer allows 10 degrees.

    python check_fit.py              # the current hookgeom
    python check_fit.py --sweep      # also try other curve starts and radii
"""

from __future__ import annotations

import math
import sys

import numpy as np
from scipy import ndimage

import hookgeom as G

HOLE_R = 0.0071 / 2  # across the flats of the 16-sided hole
MM = 1e-3


def set_curve(x0: float, r: float, deg: float) -> None:
    """Change hookgeom's top-prong curve in place (for trying other shapes)."""
    G.ARC_X0, G.ARC_R, G.ARC_DEG = x0, r, deg
    G.ARC_CENTRE = (x0, 0.0, r)
    G.TOP_TIP = (x0 + r * math.sin(math.radians(deg)), 0.0, r * (1.0 - math.cos(math.radians(deg))))


def wire_samples(step: float = 0.25 * MM, prong_only: bool = False) -> np.ndarray:
    """Centreline points (x, z) of the parts that can touch the board: top prong, back, bottom
    prong (or the top prong alone)."""
    segs = [s for s in G.segments() if min(s[0][0], s[1][0]) > -0.01]
    if prong_only:
        segs = list(zip(G.top_prong_points()[:-1], G.top_prong_points()[1:]))
    pts = []
    for a, b in segs:
        a, b = np.array([a[0], a[2]]), np.array([b[0], b[2]])
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
    return np.concatenate(pts)


def board_rects(rows: int = 3) -> np.ndarray:
    """Board material on the section: x in [0, T], z between neighbouring holes (z=0 is the top hole)."""
    out = []
    for k in range(-rows, rows + 1):
        lo = k * G.PITCH + HOLE_R
        hi = (k + 1) * G.PITCH - HOLE_R
        out.append((0.0, G.BOARD_T, lo, hi))
    return np.array(out)


RECTS = board_rects()


def clearance(xz: np.ndarray) -> np.ndarray:
    """Gap between the wire tube and the board for points (..., 2); negative inside the board."""
    x, z = xz[..., 0, None], xz[..., 1, None]
    x0, x1, z0, z1 = (RECTS[:, i] for i in range(4))
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dz = np.maximum(np.maximum(z0 - z, z - z1), 0.0)
    outside = np.hypot(dx, dz)
    inside = -np.minimum(np.minimum(x - x0, x1 - x), np.minimum(z - z0, z1 - z))  # <0 when inside
    d = np.where((dx == 0) & (dz == 0), inside, outside)
    return d.min(axis=-1) - G.WIRE_R


def pose(pts: np.ndarray, deg: float, about=(0.0, 0.0), shift=(0.0, 0.0)) -> np.ndarray:
    """Turn points arm-down by deg (negative tips the arm up) about a point, then shift them."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    u = pts - np.array(about)
    # arm points -x: arm-down is the turn that sends (-1, 0) toward (-cos, -sin)
    rot = np.array([[c, -s], [s, c]])
    return u @ rot.T + np.array(about) + np.array(shift)


def threading_margin(steps: int = 300) -> tuple[float, float, float]:
    """Smallest gap (m) between the curved prong and the board during the threading turn, the
    angle (deg) where it happens, and the smallest gap of the whole wire (the back ends the turn
    touching the face, so that one is 0 when nothing cuts)."""
    curve = wire_samples(prong_only=True)
    pts = wire_samples()
    centre = (G.ARC_CENTRE[0], G.ARC_CENTRE[2])
    worst, at, whole = 1.0, 0.0, 1.0
    for i in range(steps + 1):
        up = G.INSERT_DEG * (1 - i / steps)
        gap = float(clearance(pose(curve, -up, centre)).min())
        whole = min(whole, float(clearance(pose(pts, -up, centre)).min()))
        if gap < worst:
            worst, at = gap, up
    return worst, at, whole


def droop(max_deg: float = 20.0, dstep: float = 0.5, sstep: float = 0.15 * MM) -> float:
    """Largest arm-down tip (deg) reachable from the seated pose without entering the board."""
    pts = wire_samples(0.4 * MM)
    sx = np.arange(-9 * MM, 1.5 * MM + 1e-12, sstep)
    sz = np.arange(-3 * MM, 3 * MM + 1e-12, sstep)
    degs = np.arange(-5.0, max_deg + 1e-9, dstep)
    free = np.zeros((len(degs), len(sx), len(sz)), bool)
    for i, d in enumerate(degs):
        p = pose(pts, d)  # about the top hole's centre
        shifted = p[None, None, :, :] + np.stack(np.meshgrid(sx, sz, indexing="ij"), -1)[:, :, None, :]
        free[i] = clearance(shifted).min(axis=-1) >= 0.0
    labels, _ = ndimage.label(free)
    i0 = int(np.argmin(np.abs(degs)))
    seated = labels[i0][free[i0]]
    if seated.size == 0:
        return float("nan")
    # the seated pose: the free cell nearest no shift at 0 deg
    cells = np.argwhere(free[i0])
    j, k = cells[np.argmin(np.hypot(sx[cells[:, 0]], sz[cells[:, 1]]))]
    reach = labels == labels[i0, j, k]
    top = np.where(reach.any(axis=(1, 2)))[0].max()
    if top == len(degs) - 1:
        return float("inf")  # it can tip all the way: falls out
    return float(degs[top])


def report() -> None:
    gap, at, whole = threading_margin()
    print(f"curve from x={G.ARC_X0 * 1e3:.2f} mm, R={G.ARC_R * 1e3:.0f} mm, {G.ARC_DEG:.0f} deg: "
          f"threading margin {gap * 1e3:+.2f} mm (at {at:.1f} deg up; whole wire {whole * 1e3:+.2f}), "
          f"droop {droop():.1f} deg, "
          f"tip {G.TOP_TIP[0] * 1e3:.1f} mm behind the face, {G.TOP_TIP[2] * 1e3:.1f} mm up")


if __name__ == "__main__":
    report()
    if "--sweep" in sys.argv:
        for r in (0.015, 0.020, 0.025):
            for x0 in (0.0, 0.001, 0.002, 0.003175):
                set_curve(x0, r, G.ARC_DEG)
                report()
