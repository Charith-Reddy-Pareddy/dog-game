import numpy as np

from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_mpe import value as closed_form_value
from doggame.markov_potential import cycle_sum, value

C = DEFAULT_CONFIGS[0]
HOUSES, W = (C["house_red"], C["house_blue"]), C["w"]
START = W * np.array(HOUSES[0]) + (1 - W) * np.array(HOUSES[1])
const = lambda a: (lambda x: np.array(a, dtype=float))


def test_the_simulated_value_matches_the_closed_form_for_fixed_actions():
    red, blue = (1.0, 0.0), (0.0, 1.0)
    d_star = W * np.array(red) + (1 - W) * np.array(blue)
    for w0 in (0.0, 0.6):
        simulated = value(const(red), const(blue), HOUSES, W, w0, START)
        assert np.isclose(simulated[0], closed_form_value(START, d_star, HOUSES[0], w0, 0.9))
        assert np.isclose(simulated[1], closed_form_value(START, d_star, HOUSES[1], w0, 0.9))


def test_with_state_independent_policies_every_cycle_sums_to_zero():
    rng = np.random.default_rng(0)
    for w0 in (0.0, 0.3, 0.9):
        for _ in range(5):
            policies = [const(rng.random(2)) for _ in range(4)]
            assert abs(cycle_sum(*policies, HOUSES, W, w0, START)) < 1e-10


def test_with_policies_that_react_to_the_dog_a_cycle_does_not_sum_to_zero():
    # a counterexample: so the game is not a Markov potential game, with or without inertia
    red, blue = const((1.0, 0.0)), const((0.0, 1.0))
    red_alt, blue_alt = (lambda x: np.clip(1.0 - x, 0, 1)), (lambda x: np.clip(x, 0, 1))
    for w0 in (0.0, 0.6):
        assert abs(cycle_sum(red, red_alt, blue, blue_alt, HOUSES, W, w0, START)) > 1.0
