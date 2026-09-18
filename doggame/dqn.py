"""Discretized-action DQN / Nash-Q for the dog game.

DQN needs a finite action set, so this module discretizes the domain
into a grid (doggame.discretize) and solves the resulting general-sum
stage game with fictitious play (doggame.fictitious_play) rather than
an LP solver -- the "best response dynamics" approach the meeting
raised as the practical way to approximate a stage-game solution.

The Bellman backup for a Markov game is

    Q(s, a, b) = R(s, a, b) + discount * NashV(Q(s', ., .))

In the dog game specifically, the *next* dog position only depends on
the two actions, never on the incoming state -- so every state faces
the identical one-shot game, and the backup collapses to iterating

    Q = R + discount * NashV(Q)

on a single pair of payoff matrices until they stop moving, instead of
looping over a state space. That collapse is a property of this
particular (state-independent) transition, not a general shortcut: it
disappears as soon as the transition depends on the incoming state
(e.g. a soccer game with ball possession).
"""

import numpy as np

from doggame.fictitious_play import expected_value, fictitious_play


def solve_discretized_nash_q(
    payoff_red,
    payoff_blue,
    discount,
    tol=1e-6,
    max_outer_iter=200,
    fictitious_play_iterations=1000,
    seed=0,
):
    """Iterate the Nash-Q Bellman backup to a fixed point.

    Returns (q_red, q_blue, strategy_red, strategy_blue).
    """
    q_red = payoff_red.copy()
    q_blue = payoff_blue.copy()
    strategy_red = strategy_blue = None

    for _ in range(max_outer_iter):
        strategy_red, strategy_blue = fictitious_play(
            q_red, q_blue, iterations=fictitious_play_iterations, seed=seed
        )
        value_red = expected_value(q_red, strategy_red, strategy_blue)
        value_blue = expected_value(q_blue, strategy_red, strategy_blue)

        new_q_red = payoff_red + discount * value_red
        new_q_blue = payoff_blue + discount * value_blue

        moved = max(
            np.max(np.abs(new_q_red - q_red)), np.max(np.abs(new_q_blue - q_blue))
        )
        q_red, q_blue = new_q_red, new_q_blue
        if moved < tol:
            break

    return q_red, q_blue, strategy_red, strategy_blue
