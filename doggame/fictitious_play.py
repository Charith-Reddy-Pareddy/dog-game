"""Approximate a general-sum bimatrix game's Nash equilibrium with
fictitious play: each player repeatedly best-responds to the other's
time-averaged strategy. This is exactly the "best response dynamics"
the meeting raised as the practical way to solve a stage game without
an LP solver -- at the cost of only being approximate, and only
guaranteed to converge for some classes of games.
"""

import numpy as np


def best_response(payoff_row_player, opponent_strategy):
    """A pure best response to the opponent's (mixed) strategy, as a
    one-hot distribution over the row player's actions."""
    expected_payoffs = payoff_row_player @ opponent_strategy
    response = np.zeros(payoff_row_player.shape[0])
    response[np.argmax(expected_payoffs)] = 1.0
    return response


def fictitious_play(payoff_red, payoff_blue, iterations=2000, seed=0):
    """Iterated best response against the opponent's running average
    strategy. payoff_red[i, j] / payoff_blue[i, j] are red's / blue's
    reward when red plays action i and blue plays action j.

    Returns the two players' time-averaged (mixed) strategies.
    """
    n_red, n_blue = payoff_red.shape
    rng = np.random.default_rng(seed)

    avg_red = np.zeros(n_red)
    avg_red[rng.integers(n_red)] = 1.0
    avg_blue = np.zeros(n_blue)
    avg_blue[rng.integers(n_blue)] = 1.0

    for t in range(2, iterations + 2):
        response_red = best_response(payoff_red, avg_blue)
        response_blue = best_response(payoff_blue.T, avg_red)

        avg_red = avg_red + (response_red - avg_red) / t
        avg_blue = avg_blue + (response_blue - avg_blue) / t

    return avg_red, avg_blue


def expected_value(payoff, strategy_row, strategy_col):
    return float(strategy_row @ payoff @ strategy_col)
