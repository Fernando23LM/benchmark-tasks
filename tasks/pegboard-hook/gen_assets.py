"""Generate the OBJ meshes for the pegboard-hook task from the dimensions in task.py.

    python gen_assets.py            # writes assets/pegboard.obj, hook.obj, card.obj

Pegboard: a 500 x 350 x 6.35 mm board with real 7.1 mm round holes on a
25.4 mm grid (front/back faces triangulated cell by cell around each
16-sided hole, hole walls, and the outer rim). Hook: a 4.8 mm (3/16 in) wire swept
along its path (top prong curving up behind the board, back, bottom prong,
arm and upturned tip, plus a ball end); see hookgeom.py. Card: a 120 x 180 x 2 mm blister card
with an 8 mm hang hole. No third-party meshes.
"""

from __future__ import annotations

import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
MM = 0.001

# Board / grid (mirror task.py)
BOARD_W, BOARD_H, BOARD_T = 0.500, 0.350, 0.00635
# Holes are 16-gons 7.1 mm across the flats: their corners sit slightly outside the real circle, so
# the narrowest width is the real 9/32 in (an octagon with corners on the circle is only 6.6 mm).
HOLE_SIDES = 16
PITCH = 0.0254
COLS = range(-9, 10)  # y = m * PITCH
ROWS = range(3, 16)  # z above table = n * PITCH; board centre is 225 mm above table
BOARD_CENTRE_ABOVE_TABLE = 0.225


def _holes_local():
    out = []
    for m in COLS:
        for n in ROWS:
            out.append((m * PITCH, n * PITCH - BOARD_CENTRE_ABOVE_TABLE))
    return out


def _plate(path, thickness, cells, hole_r_flats, sides=HOLE_SIDES):
    """A watertight plate (x = thickness, faces at +-x) with one round hole per rectangular cell.

    ``cells``: (cy, cz, y0, y1, z0, z1) per hole: hole centre and the cell's bounds; the cells
    tile the plate's face. Holes are ``sides``-gons whose flats are ``hole_r_flats`` from the
    centre, so the narrowest width is the real hole diameter. Vertices are welded, so
    neighbouring cells join; the rim walls are built from the face's own boundary edges.
    """
    ht = thickness / 2
    hole_r = hole_r_flats / math.cos(math.pi / sides)
    verts, faces = [], []
    index: dict[tuple, int] = {}

    def v(x, y, z):
        key = (round(x, 8), round(y, 8), round(z, 8))
        if key not in index:
            verts.append((x, y, z))
            index[key] = len(verts)  # OBJ is 1-based
        return index[key]

    def tri(a, b, c):
        faces.append((a, b, c))

    for (cy, cz, y0, y1, z0, z1) in cells:
        rings = {}
        for side, x in (("f", -ht), ("b", ht)):
            inner, outer = [], []
            for k in range(sides):
                a = 2 * math.pi * (k + 0.5) / sides
                d = (math.cos(a), math.sin(a))  # (y, z) direction
                inner.append(v(x, cy + hole_r * d[0], cz + hole_r * d[1]))
                # ray from the hole centre to the cell's boundary
                t = math.inf
                if d[0] > 1e-9:
                    t = min(t, (y1 - cy) / d[0])
                elif d[0] < -1e-9:
                    t = min(t, (y0 - cy) / d[0])
                if d[1] > 1e-9:
                    t = min(t, (z1 - cz) / d[1])
                elif d[1] < -1e-9:
                    t = min(t, (z0 - cz) / d[1])
                outer.append(v(x, cy + t * d[0], cz + t * d[1]))
            # each cell corner lies between two rays: fill it with its own triangle
            corner_at = {}
            for (py, pz) in ((y1, z1), (y0, z1), (y0, z0), (y1, z0)):
                ang = math.atan2(pz - cz, py - cy) % (2 * math.pi)
                corner_at[int(ang / (2 * math.pi / sides) - 0.5) % sides] = v(x, py, pz)
            for k in range(sides):
                k2 = (k + 1) % sides
                if side == "f":  # front faces -x
                    tri(outer[k], inner[k], inner[k2])
                    tri(outer[k], inner[k2], outer[k2])
                    if k in corner_at:
                        tri(outer[k], outer[k2], corner_at[k])
                else:
                    tri(outer[k], inner[k2], inner[k])
                    tri(outer[k], outer[k2], inner[k2])
                    if k in corner_at:
                        tri(outer[k], corner_at[k], outer[k2])
            rings[side] = inner
        fi, bi = rings["f"], rings["b"]
        for k in range(sides):  # hole wall, facing the hole's axis
            k2 = (k + 1) % sides
            tri(fi[k], bi[k], bi[k2])
            tri(fi[k], bi[k2], fi[k2])
    # Rim walls: a front edge p->q with no partner q->p lies on the rim; the wall runs to the
    # matching back-face vertices, wound against the front edge.
    directed = {(f[i], f[(i + 1) % 3]) for f in faces for i in range(3)}
    for a, b in sorted(directed):
        if (b, a) in directed:
            continue
        pa, pb = verts[a - 1], verts[b - 1]
        if abs(pa[0] + ht) > 1e-9 or abs(pb[0] + ht) > 1e-9:
            continue
        ba, bb = v(ht, pa[1], pa[2]), v(ht, pb[1], pb[2])
        tri(b, a, ba)
        tri(b, ba, bb)
    _write(path, verts, faces)
    return len(faces)


def gen_pegboard(path):
    """Cells: halfway to the neighbouring holes; the outermost cells stretch to the board's edge."""
    holes = _holes_local()
    hw, hh = BOARD_W / 2, BOARD_H / 2
    ys = sorted({y for y, _ in holes})
    zs = sorted({z for _, z in holes})

    def edges(centres, c, edge):
        i = centres.index(c)
        lo = -edge if i == 0 else (centres[i - 1] + c) / 2
        hi = edge if i == len(centres) - 1 else (c + centres[i + 1]) / 2
        return lo, hi

    cells = [(cy, cz, *edges(ys, cy, hw), *edges(zs, cz, hh)) for cy, cz in holes]
    return len(holes), _plate(path, BOARD_T, cells, 0.0071 / 2)


def _tube(verts, faces, p1, p2, r, n=10):
    a, b = _vec(p1), _vec(p2)
    axis = b - a
    axis = axis / _norm(axis)
    ref = _vec((0.0, 1.0, 0.0)) if abs(axis[1]) < 0.9 else _vec((1.0, 0.0, 0.0))
    u = _cross(axis, ref); u = u / _norm(u)
    w = _cross(axis, u)
    r1, r2 = [], []
    for k in range(n):
        t = 2 * math.pi * k / n
        d = u * math.cos(t) + w * math.sin(t)
        r1.append(_add_v(verts, a + r * d))
        r2.append(_add_v(verts, b + r * d))
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((r1[k], r2[k2], r2[k]))  # outward-facing sides
        faces.append((r1[k], r1[k2], r2[k2]))
    # end caps (fan to centres)
    c1, c2 = _add_v(verts, a), _add_v(verts, b)
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((c1, r1[k2], r1[k]))
        faces.append((c2, r2[k], r2[k2]))


def _sphere(verts, faces, centre, r, n=10, m=6):
    c = _vec(centre)
    rings = []
    for j in range(m + 1):
        phi = math.pi * j / m
        ring = []
        for k in range(n):
            t = 2 * math.pi * k / n
            d = _vec((math.sin(phi) * math.cos(t), math.sin(phi) * math.sin(t), math.cos(phi)))
            ring.append(_add_v(verts, c + r * d))
        rings.append(ring)
    for j in range(m):
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((rings[j][k], rings[j + 1][k], rings[j + 1][k2]))
            faces.append((rings[j][k], rings[j + 1][k2], rings[j][k2]))


def _vec(t):
    import numpy as np
    return np.array(t, float)


def _norm(a):
    import numpy as np
    return float(np.linalg.norm(a))


def _cross(a, b):
    import numpy as np
    return np.cross(a, b)


def _add_v(verts, p):
    verts.append(tuple(float(x) for x in p))
    return len(verts)


# Hook wire path: see hookgeom.py (shared with task.py and demo.py).
import hookgeom as G  # noqa: E402

WIRE_R = G.WIRE_R


def gen_hook(path):
    verts, faces = [], []
    segs = G.segments()
    for p1, p2 in segs:
        _tube(verts, faces, p1, p2, WIRE_R, n=12)
    # round every joint so the swept wire has no notches (the curve is a chain of short pieces)
    joints = {p for seg in segs for p in seg} - {G.TIP_END, G.TOP_TIP}
    for p in sorted(joints):
        _sphere(verts, faces, p, WIRE_R)
    _sphere(verts, faces, G.TOP_TIP, WIRE_R)  # rounded prong tip
    _sphere(verts, faces, G.TIP_END, G.BALL_R)  # 6 mm ball end
    _write(path, verts, faces)
    return len(faces)


CARD_T = 0.002
CARD_HOLE_R = 0.004


def gen_card(path, card_w, card_h, hole_off):
    """A blister card: a thin plate with one 8 mm hang hole ``hole_off`` above its centre."""
    cell = (0.0, hole_off, -card_w / 2, card_w / 2, -card_h / 2, card_h / 2)
    return _plate(path, CARD_T, [cell], CARD_HOLE_R)


def _write(path, verts, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for p in verts:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")


if __name__ == "__main__":
    n_holes, n_board = gen_pegboard(HERE / "assets" / "pegboard.obj")
    n_hook = gen_hook(HERE / "assets" / "hook.obj")
    n_card = gen_card(HERE / "assets" / "card.obj", 0.120, 0.120, 0.045)
    gen_card(HERE / "assets" / "card_small.obj", 0.100, 0.045, 0.0145)
    print(f"pegboard: {n_holes} holes, {n_board} tris; hook: {n_hook} tris; card: {n_card} tris")
