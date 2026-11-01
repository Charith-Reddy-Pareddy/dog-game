"""Archived tests: frozen copies removed from tests/test_verify.py. They are not imported or tested here; run them at the
git tag pre-archive."""


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
