import numpy as np

from doggame.discretize import action_grid, payoff_matrices
from doggame.dqn import solve_discretized_nash_q
from doggame.env import Square, negative_squared_distance_reward, weighted_average_transition
from doggame.fictitious_play import expected_value
from doggame.nash import solve_stage_nash


def build_game(resolution=9):
    domain = Square(0.0, 1.0)
    house_red = np.array([0.9, 0.2])
    house_blue = np.array([0.1, 0.8])
    w = 0.5
    grid = action_grid(domain, resolution)
    payoff_red, payoff_blue = payoff_matrices(
        grid,
        house_red,
        house_blue,
        w,
        domain,
        weighted_average_transition,
        negative_squared_distance_reward,
    )
    return domain, house_red, house_blue, w, grid, payoff_red, payoff_blue


def test_discretized_equilibrium_concentrates_near_the_continuous_nash():
    domain, house_red, house_blue, w, grid, payoff_red, payoff_blue = build_game()

    _, _, strategy_red, strategy_blue = solve_discretized_nash_q(
        payoff_red, payoff_blue, discount=0.9
    )

    nash_red, nash_blue, _ = solve_stage_nash(house_red, house_blue, w, domain)

    picked_red = grid[np.argmax(strategy_red)]
    picked_blue = grid[np.argmax(strategy_blue)]

    assert np.allclose(picked_red, nash_red, atol=1e-6)
    assert np.allclose(picked_blue, nash_blue, atol=1e-6)
    # both strategies should be (close to) pure, since the continuous
    # equilibrium here is a single boundary point, not a mix
    assert strategy_red.max() > 0.99
    assert strategy_blue.max() > 0.99


def test_q_values_match_the_closed_form_fixed_point():
    # Because the dog game's transition ignores the incoming state,
    # the Nash-Q fixed point has a closed form: discounted value =
    # (one-shot Nash value) / (1 - discount). Solving via repeated
    # Bellman backups should land on exactly that.
    _, _, _, _, _, payoff_red, payoff_blue = build_game(resolution=5)
    discount = 0.8

    q_red, q_blue, strategy_red, strategy_blue = solve_discretized_nash_q(
        payoff_red, payoff_blue, discount=discount
    )

    stage_value_red = expected_value(payoff_red, strategy_red, strategy_blue)
    stage_value_blue = expected_value(payoff_blue, strategy_red, strategy_blue)

    expected_discounted_red = stage_value_red / (1 - discount)
    expected_discounted_blue = stage_value_blue / (1 - discount)

    actual_discounted_red = expected_value(q_red, strategy_red, strategy_blue)
    actual_discounted_blue = expected_value(q_blue, strategy_red, strategy_blue)

    assert np.isclose(actual_discounted_red, expected_discounted_red, rtol=1e-3)
    assert np.isclose(actual_discounted_blue, expected_discounted_blue, rtol=1e-3)
