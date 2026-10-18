import numpy as np

from doggame.exact_play import GRID, best_actions, play, score
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


def test_the_many_player_check_scores_the_target_form_at_zero_and_holding_above_it():
    from doggame.exact_play import deviation_gain_many, play_many
    from doggame.many_houses import one_shot, target_equilibrium

    houses, weights, w0 = np.array([0.8, 0.45, 0.1]), np.array([0.3, 0.3, 0.4]), 0.6
    targets, _ = target_equilibrium(houses, weights, w0)
    xs = np.linspace(0, 1, GRID)
    target_form = np.stack([one_shot((targets - w0 * x) / (1 - w0), weights)[0] for x in xs], 1)
    holding = np.tile(one_shot(houses, weights)[0][:, None], (1, GRID))
    small = dict(states=6, grid=51, rounds=80)
    assert deviation_gain_many(target_form, houses, weights, w0, **small) < 1e-4 < deviation_gain_many(holding, houses, weights, w0, **small)
    assert play_many(houses, weights, w0, steps=3, rounds=10).shape == (3, GRID)


def test_best_responses_to_proposition_2_are_its_own_actions():
    grid = np.linspace(0, 1, GRID)
    a_red, a_blue, _ = nash_1d(*SETTING[:3])
    red, blue = explicit_policy(*SETTING[:3], a_red, a_blue, SETTING[3])(grid)
    best = best_actions(red, blue, *SETTING, actions=101, rounds=60)
    inner = slice(2, -2)  # away from the edges, where actions tie
    assert np.abs(best[0][inner] - red[inner]).max() < 0.02 and np.abs(best[1][inner] - blue[inner]).max() < 0.02
