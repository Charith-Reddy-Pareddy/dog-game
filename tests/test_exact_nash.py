import numpy as np

from doggame.discretize import action_grid, payoff_matrices
from doggame.env import Square, negative_squared_distance_reward, weighted_average_transition
from doggame.exact_nash import lemke_howson_nash, zero_sum_lp_nash
from doggame.nash import solve_stage_nash


def test_lemke_howson_matches_matching_pennies():
    payoff_red = np.array([[1.0, -1.0], [-1.0, 1.0]])
    payoff_blue = -payoff_red

    strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)

    assert np.allclose(strategy_red, [0.5, 0.5])
    assert np.allclose(strategy_blue, [0.5, 0.5])


def test_lemke_howson_on_the_planning_notes_example_matrix():
    # the 3x3 example matrix from the planning notes, treated as a
    # zero-sum game (row player's payoff is Q, column player's is -Q)
    payoff_red = np.array(
        [
            [10.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
        ]
    )
    payoff_blue = -payoff_red

    strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)

    # a valid probability distribution over each player's 3 actions
    assert strategy_red.shape == (3,)
    assert strategy_blue.shape == (3,)
    assert np.isclose(strategy_red.sum(), 1.0)
    assert np.isclose(strategy_blue.sum(), 1.0)
    assert np.all(strategy_red >= -1e-9)
    assert np.all(strategy_blue >= -1e-9)

    # it should actually be an equilibrium: neither side can do better
    # by unilaterally switching to any single pure action
    value_red = strategy_red @ payoff_red @ strategy_blue
    for i in range(3):
        pure_i = np.eye(3)[i]
        assert pure_i @ payoff_red @ strategy_blue <= value_red + 1e-6
    value_blue = strategy_red @ payoff_blue @ strategy_blue
    for j in range(3):
        pure_j = np.eye(3)[j]
        assert strategy_red @ payoff_blue @ pure_j <= value_blue + 1e-6


def test_lemke_howson_matches_the_discretized_dog_game_corner_solution():
    domain = Square(0.0, 1.0)
    house_red, house_blue, w = (0.9, 0.2), (0.1, 0.8), 0.5
    grid = action_grid(domain, resolution=5)
    payoff_red, payoff_blue = payoff_matrices(
        grid, house_red, house_blue, w, domain,
        weighted_average_transition, negative_squared_distance_reward,
    )

    strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)

    nash_red, nash_blue, _ = solve_stage_nash(house_red, house_blue, w, domain)
    picked_red = grid[np.argmax(strategy_red)]
    picked_blue = grid[np.argmax(strategy_blue)]

    assert np.allclose(picked_red, nash_red, atol=1e-6)
    assert np.allclose(picked_blue, nash_blue, atol=1e-6)
    # this stage game's equilibrium is a single boundary point, so the
    # exact solver should find it as a pure (not mixed) strategy
    assert strategy_red.max() > 0.99
    assert strategy_blue.max() > 0.99


def test_zero_sum_lp_matches_matching_pennies():
    payoff_red = np.array([[1.0, -1.0], [-1.0, 1.0]])

    strategy_red, strategy_blue, value = zero_sum_lp_nash(payoff_red)

    assert np.allclose(strategy_red, [0.5, 0.5], atol=1e-6)
    assert np.allclose(strategy_blue, [0.5, 0.5], atol=1e-6)
    assert abs(value) < 1e-6


def test_zero_sum_lp_matches_rock_paper_scissors():
    payoff_red = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    strategy_red, strategy_blue, value = zero_sum_lp_nash(payoff_red)

    assert np.allclose(strategy_red, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert np.allclose(strategy_blue, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)
    assert abs(value) < 1e-6


def test_zero_sum_lp_finds_a_pure_dominant_strategy():
    # red's first action strictly dominates: it always beats blue's
    # first action and always at least draws against the second
    payoff_red = np.array([[1.0, 0.5], [-1.0, 0.0]])

    strategy_red, _, value = zero_sum_lp_nash(payoff_red)

    assert np.isclose(strategy_red[0], 1.0, atol=1e-6)
    assert value > 0.0
