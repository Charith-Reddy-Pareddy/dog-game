import numpy as np

from doggame.fictitious_play import best_response_dynamics, expected_value, fictitious_play


def test_converges_to_the_unique_pure_equilibrium():
    # A trivial coordination-free game: red always prefers action 0,
    # blue always prefers action 1, regardless of what the other does.
    payoff_red = np.array([[1.0, 1.0], [0.0, 0.0]])
    payoff_blue = np.array([[0.0, 1.0], [0.0, 1.0]])

    strategy_red, strategy_blue = fictitious_play(payoff_red, payoff_blue, iterations=200)

    # the running average decays as 1/t, so it approaches but never
    # exactly reaches the pure equilibrium
    assert np.allclose(strategy_red, [1.0, 0.0], atol=0.01)
    assert np.allclose(strategy_blue, [0.0, 1.0], atol=0.01)


def test_matching_pennies_converges_to_the_known_mixed_equilibrium():
    # Classic zero-sum textbook example: red wants to match, blue
    # wants to mismatch. The unique Nash equilibrium mixes 50/50.
    payoff_red = np.array([[1.0, -1.0], [-1.0, 1.0]])
    payoff_blue = -payoff_red

    strategy_red, strategy_blue = fictitious_play(
        payoff_red, payoff_blue, iterations=20_000, seed=1
    )

    assert np.allclose(strategy_red, [0.5, 0.5], atol=0.05)
    assert np.allclose(strategy_blue, [0.5, 0.5], atol=0.05)

    # at the equilibrium, the game's value is 0 for both players
    assert abs(expected_value(payoff_red, strategy_red, strategy_blue)) < 0.05
    assert abs(expected_value(payoff_blue, strategy_red, strategy_blue)) < 0.05


ROCK_PAPER_SCISSORS = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def test_best_response_dynamics_cycle_in_rock_paper_scissors():
    history = best_response_dynamics(ROCK_PAPER_SCISSORS, -ROCK_PAPER_SCISSORS, steps=30)
    red_moves = [red for red, _ in history]

    # it never settles on one action, it keeps cycling through all three
    assert all(a != b for a, b in zip(red_moves, red_moves[1:]))
    assert set(red_moves) == {0, 1, 2}


def test_fictitious_play_settles_on_one_third_each_in_rock_paper_scissors():
    strategy_red, strategy_blue = fictitious_play(
        ROCK_PAPER_SCISSORS, -ROCK_PAPER_SCISSORS, iterations=20_000, seed=1
    )

    assert np.allclose(strategy_red, [1 / 3] * 3, atol=0.05)
    assert np.allclose(strategy_blue, [1 / 3] * 3, atol=0.05)
