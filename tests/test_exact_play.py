import numpy as np

from doggame.exact_play import GRID, play, score
from doggame.inertia_theory import explicit_policy, nash_1d

SETTING = (0.7, 0.6, 0.7, 0.6)  # setup 3's x coordinate at inertia 0.6


def test_the_score_is_zero_for_proposition_2_and_positive_for_holding():
    grid = np.linspace(0, 1, GRID)
    a_red, a_blue, _ = nash_1d(*SETTING[:3])
    exact = score(*explicit_policy(*SETTING[:3], a_red, a_blue, SETTING[3])(grid), *SETTING)
    hold = score(np.full(GRID, a_red), np.full(GRID, a_blue), *SETTING)
    assert exact["gap_explicit"] < 1e-12 and exact["gain"] < 1e-9
    assert hold["gap_holding"] == 0 and hold["gain"] > 0.05


def test_a_table_of_actions_moves_toward_the_feedback_equilibrium():
    red, blue = play(*SETTING, kind="table", squash="clamp", steps=400, rounds=40)
    result = score(red, blue, *SETTING)
    assert result["gap_explicit"] < 0.1 < result["gap_holding"]
