"""Synthetic end states for every case in the task's Scoring text.
Run: python -m pytest test_scoring.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from scoring import EndState, HookMeasure, is_seated, score  # noqa: E402

SEATED = HookMeasure(top_in_tagged=True, top_tip_behind=True, bottom_depth_mm=8.0,
                     back_gap_mm=0.0, arm_square_deg=0.0)
TOP_ONLY = HookMeasure(top_in_tagged=True, top_tip_behind=True, bottom_depth_mm=0.0,
                       back_gap_mm=6.0, arm_square_deg=35.0)
NONE = HookMeasure()


def end(a=SEATED, b=SEATED, **kw):
    return EndState(hooks=(a, b), **kw)


def test_both_seated_and_released_is_full_score():
    assert score(end(clear_s=1.0)) == 1.0


def test_both_seated_gripper_still_touching():
    assert score(end(clear_s=0.0)) == 0.8


def test_both_seated_released_for_less_than_a_second():
    assert score(end(clear_s=0.95)) == 0.8


def test_a_seated_b_top_only():
    assert score(end(b=TOP_ONLY, clear_s=5.0)) == 0.55


def test_a_seated_at_untagged_hole_b_seated():
    # Seated somewhere else: measured against its tagged hole it is not
    # top-engaged at all (top_in_tagged is False), so it earns nothing.
    elsewhere = HookMeasure(top_in_tagged=False, top_tip_behind=True, bottom_depth_mm=8.0,
                            back_gap_mm=0.0, arm_square_deg=0.0)
    assert score(end(a=elsewhere, clear_s=5.0)) == 0.4


def test_both_top_only():
    assert score(end(a=TOP_ONLY, b=TOP_ONLY, clear_s=5.0)) == 0.3


def test_back_three_mm_off_is_not_seated():
    off = HookMeasure(**{**SEATED.__dict__, "back_gap_mm": 3.0})
    assert not is_seated(off)
    # It still has its top prong in, so it earns only the top-engaged points.
    assert score(end(a=off, b=NONE)) == 0.15


def test_arm_twelve_degrees_off_is_not_seated():
    off = HookMeasure(**{**SEATED.__dict__, "arm_square_deg": 12.0})
    assert not is_seated(off)
    assert score(end(a=off, b=NONE)) == 0.15


def test_arm_sagging_eight_degrees_is_still_seated():
    # A seated 3/16 in hook sags a few degrees in its 7.1 mm holes; up to 10 deg counts as square.
    assert is_seated(HookMeasure(**{**SEATED.__dict__, "arm_square_deg": 8.0}))


def test_bottom_four_mm_in_is_not_seated():
    shallow = HookMeasure(**{**SEATED.__dict__, "bottom_depth_mm": 4.0})
    assert not is_seated(shallow)


def test_bottom_five_mm_in_is_seated():
    assert is_seated(HookMeasure(**{**SEATED.__dict__, "bottom_depth_mm": 5.0}))


def test_product_fallen_is_zero():
    assert score(end(clear_s=5.0, product_fallen=True)) == 0.0


def test_existing_hook_pulled_out_is_zero():
    assert score(end(clear_s=5.0, existing_knocked=True)) == 0.0


def test_new_hook_on_the_floor_is_zero():
    assert score(end(clear_s=5.0, new_off_table=True)) == 0.0


def test_nothing_done():
    assert score(EndState()) == 0.0


def test_only_listed_scores_are_possible():
    for a in (SEATED, TOP_ONLY, NONE):
        for b in (SEATED, TOP_ONLY, NONE):
            for clear in (0.0, 1.0):
                assert score(end(a=a, b=b, clear_s=clear)) in (0.0, 0.15, 0.3, 0.4, 0.55, 0.8, 1.0)
