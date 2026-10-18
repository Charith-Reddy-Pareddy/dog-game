import numpy as np

from doggame.duopoly_learning import GRID, best_outputs, score
from doggame.sticky_duopoly import A, Q_MAX, linear_equilibrium


def test_the_feedback_equilibrium_scores_zero_and_is_its_own_best_response():
    a, b, _ = linear_equilibrium(0.8)
    prices = np.linspace(0, A, GRID)
    outputs = np.clip(a + b * prices, 0, Q_MAX)
    assert score(np.stack([outputs, outputs]), 0.8, grid=201, rounds=120)["gain"] < 1e-4
    policy = [lambda p: np.interp(p, prices, outputs)] * 2
    best = best_outputs(policy, 0.8)
    assert np.abs(best - outputs)[:, 2:-2].max() < 0.01


def test_repeating_cournot_is_not_a_feedback_equilibrium_with_sticky_prices():
    cournot = np.full((2, GRID), 0.225)
    assert score(cournot, 0.8, grid=201, rounds=120)["gain"] > 1e-3
