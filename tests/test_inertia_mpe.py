import itertools

import numpy as np

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_mpe import check_mpe, one_shot_gain, value
from doggame.nash import solve_stage_nash


def nash(config):
    return tuple(solve_stage_nash(config["house_red"], config["house_blue"], config["w"], Square(0.0, 1.0))[:2])


def simulate(x, house, d_star, w0, discount, rounds=600):
    total, x = 0.0, np.asarray(x, dtype=float)
    for k in range(rounds):
        x = w0 * x + (1 - w0) * d_star
        total -= discount ** k * np.sum((x - house) ** 2)
    return total


def test_the_closed_form_value_matches_a_simulation():
    house, d_star = np.array([0.9, 0.2]), np.array([0.5, 0.5])
    for w0 in (0.0, 0.3, 0.9):
        for x in ((0.1, 0.9), (0.7, 0.3)):
            assert np.isclose(value(x, d_star, house, w0, 0.9), simulate(x, house, d_star, w0, 0.9))


def test_the_exact_one_round_gain_matches_a_brute_force_search_over_actions():
    c, w0, g = DEFAULT_CONFIGS[0], 0.6, 0.9
    house, own, other = np.array(c["house_red"]), np.array([0.5, 0.5]), np.array(nash(c)[1])
    x = np.array([0.3, 0.7])
    d_star = c["w"] * own + (1 - c["w"]) * other
    q = lambda a: (-np.sum((w0 * x + (1 - w0) * (c["w"] * np.array(a) + (1 - c["w"]) * other) - house) ** 2)
                   + g * value(w0 * x + (1 - w0) * (c["w"] * np.array(a) + (1 - c["w"]) * other), d_star, house, w0, g))
    brute = max(q(a) for a in itertools.product(np.linspace(0, 1, 201), repeat=2)) - q(own)

    gain, _ = one_shot_gain(x, house, own, other, c["w"], w0, g)
    assert gain > 1e-3 and abs(gain - brute) < 1e-4


def test_holding_the_one_shot_nash_actions_is_a_feedback_equilibrium_without_inertia():
    c = DEFAULT_CONFIGS[0]
    result = check_mpe((c["house_red"], c["house_blue"]), c["w"], 0.0, nash(c), grid=11)

    assert result["red"]["max_gain"] < 1e-12 and result["blue"]["max_gain"] < 1e-12


def test_a_profile_that_is_not_an_equilibrium_shows_a_gain():
    c = DEFAULT_CONFIGS[0]
    result = check_mpe((c["house_red"], c["house_blue"]), c["w"], 0.6, ((0.5, 0.5), nash(c)[1]), grid=11)

    assert result["red"]["max_gain"] > 1e-3 and result["red"]["start_gain"] > 1e-3
