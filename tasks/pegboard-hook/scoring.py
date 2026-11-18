"""Scoring for the pegboard-hook task. Pure Python: no simulator needed.

The simulator (or a person with a ruler and a photo) fills in a
:class:`HookMeasure` for each new hook and an :class:`EndState`;
:func:`score` turns them into the trial's score. Every threshold in the
task's Scoring text is a constant here.

A hook is **top-engaged** when its top prong is in its tagged hole with
the bent tip behind the board. It is **seated** when, in addition, its
bottom prong is at least ``BOTTOM_MIN_MM`` into the hole below, its back
is within ``BACK_FLUSH_MM`` of the board face, and its arm is square
within ``ARM_SQUARE_DEG`` seen from above and from the side. In
simulation the 20 N pull check is replaced by this prong-engagement
geometry (see README).

Points: ``TOP_POINTS`` per top-engaged hook, or ``SEATED_POINTS`` per
seated hook in place of that, plus ``RELEASE_POINTS`` when both hooks
are seated and both grippers have been clear of both new hooks for the
last ``CLEAR_S`` seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

BOTTOM_MIN_MM = 5.0  # bottom prong at least this far into the hole below
BACK_FLUSH_MM = 2.0  # back within this of the board face
ARM_SQUARE_DEG = 10.0  # arm within this of square, seen from above and from the side (a seated 3/16 in hook sags 5-9 deg in its holes)
CLEAR_S = 1.0  # both grippers clear of both new hooks for this long
CLEAR_MM = 20.0  # a gripper is clear at this distance from a new hook

TOP_POINTS = 0.15
SEATED_POINTS = 0.4
RELEASE_POINTS = 0.2


@dataclass(frozen=True)
class HookMeasure:
    """One new hook measured against its tagged hole at the end."""

    top_in_tagged: bool = False  # top prong passes through the tagged hole
    top_tip_behind: bool = False  # bent tip behind the board's back face
    bottom_depth_mm: float = 0.0  # bottom prong depth into the hole below
    back_gap_mm: float = 99.0  # gap between the back and the board face
    arm_square_deg: float = 90.0  # arm off square: the larger of its angle seen from above and from the side


@dataclass(frozen=True)
class EndState:
    hooks: tuple[HookMeasure, HookMeasure] = (HookMeasure(), HookMeasure())  # (A, B)
    clear_s: float = 0.0  # both grippers clear of both new hooks, up to the end
    product_fallen: bool = False  # a blister card off its existing hook at any/end check
    existing_knocked: bool = False  # an existing hook out of the board (top prong out of its hole)
    new_off_table: bool = False  # a new hook more than 2 cm below the table top


def is_top_engaged(m: HookMeasure) -> bool:
    return bool(m.top_in_tagged and m.top_tip_behind)


def is_seated(m: HookMeasure) -> bool:
    return bool(
        is_top_engaged(m)
        and m.bottom_depth_mm >= BOTTOM_MIN_MM
        and m.back_gap_mm <= BACK_FLUSH_MM
        and m.arm_square_deg <= ARM_SQUARE_DEG
    )


def score(state: EndState) -> float:
    if state.product_fallen or state.existing_knocked or state.new_off_table:
        return 0.0
    total = 0.0
    for m in state.hooks:
        if is_seated(m):
            total += SEATED_POINTS
        elif is_top_engaged(m):
            total += TOP_POINTS
    if all(is_seated(m) for m in state.hooks) and state.clear_s >= CLEAR_S:
        total += RELEASE_POINTS
    return round(total, 3)
