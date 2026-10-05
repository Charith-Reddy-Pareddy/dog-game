"""Exact solution of the soccer game by minimax value iteration.

Every state is a zero-sum 4x4 matrix game, Q(s, a0, a1). Value iteration
sets Q = reward + discount * V(next state) and V(s) to the value of that
matrix game, until V stops moving. Games with a saddle point (a pure
equilibrium) are read straight off the matrix; the rest need a mixed
strategy and are solved as a linear program.
"""

from collections import namedtuple

import numpy as np

from doggame.exact_nash import zero_sum_lp_nash

Solution = namedtuple("Solution", "value q policy0 policy1 mixed")


def _matrix_game(q):
    """Value and both players' strategies for every state's matrix game.
    `mixed[s]` is True when state s has no saddle point."""
    n, k, _ = q.shape
    lower = q.min(axis=2)  # worst case for player 0 of each row
    upper = q.max(axis=1)  # worst case for player 1 of each column
    best_row, best_col = lower.argmax(axis=1), upper.argmin(axis=1)
    value = lower.max(axis=1)
    mixed = ~np.isclose(value, upper.min(axis=1))

    policy0 = np.eye(k)[best_row]
    policy1 = np.eye(k)[best_col]
    for s in np.flatnonzero(mixed):
        policy0[s], policy1[s], value[s] = zero_sum_lp_nash(q[s])
    return value, policy0, policy1, mixed


def solve_soccer(game, discount=0.9, tol=1e-6, max_iter=500):
    value = np.zeros(game.n_states + 1)  # the extra entry is the finished game, worth 0
    for _ in range(max_iter):
        q = game.reward + discount * value[game.next_state]
        new_value, _, _, _ = _matrix_game(q)
        change = np.abs(new_value - value[:-1]).max()
        value[:-1] = new_value
        if change < tol:
            break

    q = game.reward + discount * value[game.next_state]
    value[:-1], policy0, policy1, mixed = _matrix_game(q)
    return Solution(value[:-1], q, policy0, policy1, mixed)
