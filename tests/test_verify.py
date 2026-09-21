import numpy as np

from doggame.discretize import action_grid, payoff_matrices
from doggame.dqn import solve_discretized_nash_q
from doggame.env import Square, negative_squared_distance_reward, weighted_average_transition
from doggame.nash import solve_stage_nash
from doggame.verify import exploitability, mixed_strategy_indifference_gap


def test_exploitability_is_near_zero_at_the_analytical_nash():
    domain = Square(0.0, 1.0)
    house_red, house_blue, w = (0.9, 0.3), (0.1, 0.7), 0.4
    action_red, action_blue, _ = solve_stage_nash(house_red, house_blue, w, domain)

    red_gap, blue_gap = exploitability(
        action_red, action_blue, house_red, house_blue, w, domain, resolution=201
    )

    assert red_gap < 1e-3
    assert blue_gap < 1e-3


def test_exploitability_is_clearly_positive_for_a_naive_non_equilibrium():
    # each player naively states their true house, with no
    # exaggeration to counteract the other -- not a Nash equilibrium
    domain = Square(0.0, 1.0)
    house_red, house_blue, w = (0.9, 0.3), (0.1, 0.7), 0.4

    red_gap, blue_gap = exploitability(
        house_red, house_blue, house_red, house_blue, w, domain
    )

    assert red_gap > 0.05
    assert blue_gap > 0.05


def test_mixed_strategy_indifference_gap_is_small_for_a_solved_stage_game():
    domain = Square(0.0, 1.0)
    house_red, house_blue, w = (0.9, 0.2), (0.1, 0.8), 0.5
    grid = action_grid(domain, resolution=9)
    payoff_red, payoff_blue = payoff_matrices(
        grid,
        house_red,
        house_blue,
        w,
        domain,
        weighted_average_transition,
        negative_squared_distance_reward,
    )
    _, _, strategy_red, strategy_blue = solve_discretized_nash_q(
        payoff_red, payoff_blue, discount=0.9
    )

    spread_red, gap_red = mixed_strategy_indifference_gap(
        payoff_red, strategy_red, strategy_blue
    )
    spread_blue, gap_blue = mixed_strategy_indifference_gap(
        payoff_blue.T, strategy_blue, strategy_red
    )

    assert spread_red < 1e-6
    assert gap_red < 1e-6
    assert spread_blue < 1e-6
    assert gap_blue < 1e-6


def test_indifference_gap_flags_mixing_between_unequal_actions():
    # playing two actions of clearly different value with positive
    # probability violates the indifference condition
    payoff_red = np.array([[1.0, 1.0], [0.0, 0.0]])
    mixed = np.array([0.5, 0.5])
    opponent = np.array([0.5, 0.5])

    spread, _ = mixed_strategy_indifference_gap(payoff_red, mixed, opponent)

    assert spread > 0.4


def test_indifference_gap_flags_a_dominated_pure_strategy():
    # playing only the strictly worse action should show a clear
    # best-response gap to the unplayed, better action
    payoff_red = np.array([[1.0, 1.0], [0.0, 0.0]])
    dominated = np.array([0.0, 1.0])
    opponent = np.array([0.5, 0.5])

    _, gap = mixed_strategy_indifference_gap(payoff_red, dominated, opponent)

    assert gap > 0.4


def test_indifference_gap_stays_finite_for_a_uniform_strategy_on_a_fine_grid():
    # On a grid finer than 1/tol actions, a genuinely uniform mixed
    # strategy puts every action below the "played" probability floor.
    # The gap must still come back finite (and ~0, since a uniform
    # strategy against a matching-pennies-style payoff is indifferent
    # across every action) rather than +inf.
    n = 1500
    payoff_red = np.array([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
    strategy_row = np.full(n, 1.0 / n)
    strategy_opponent = np.full(n, 1.0 / n)

    spread, gap = mixed_strategy_indifference_gap(payoff_red, strategy_row, strategy_opponent)

    assert np.isfinite(spread)
    assert np.isfinite(gap)
