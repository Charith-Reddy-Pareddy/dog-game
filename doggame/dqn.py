"""Discretized-action DQN / Nash-Q for the dog game.

DQN needs a finite action set, so this module discretizes the domain
into a grid (doggame.discretize) and solves the resulting general-sum
stage game with fictitious play (doggame.fictitious_play) by default --
the "best response dynamics" approach the meeting raised as the
practical way to approximate a stage-game solution -- or, via
`solver="exact"`, with NashPy's Lemke-Howson algorithm
(doggame.exact_nash), which is exact rather than approximate.

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

Re-solving the stage game's Nash equilibrium on *every* outer Bellman
iteration is wasteful once Q is close to converged, since the
equilibrium strategy barely moves between iterations near the fixed
point. `resync_every` implements the cheaper schedule from the
planning notes instead: solve the stage game once, freeze that
strategy for `resync_every` outer iterations of plain Q-value updates
under it, then re-solve ("sync") and repeat. `resync_every=1` (the
default) recomputes every iteration, identical to the original
behavior.
"""

import numpy as np

from doggame.exact_nash import lemke_howson_nash
from doggame.fictitious_play import expected_value, fictitious_play

SOLVERS = {
    "fictitious_play": lambda q_red, q_blue, fictitious_play_iterations, seed: fictitious_play(
        q_red, q_blue, iterations=fictitious_play_iterations, seed=seed
    ),
    "exact": lambda q_red, q_blue, fictitious_play_iterations, seed: lemke_howson_nash(q_red, q_blue),
}


def solve_discretized_nash_q(
    payoff_red,
    payoff_blue,
    discount,
    tol=1e-6,
    max_outer_iter=200,
    fictitious_play_iterations=1000,
    seed=0,
    solver="fictitious_play",
    resync_every=1,
):
    """Iterate the Nash-Q Bellman backup to a fixed point.

    `solver` selects how each stage game is solved: "fictitious_play"
    (approximate, the default) or "exact" (NashPy's Lemke-Howson).
    `resync_every` controls how many outer iterations reuse a frozen
    strategy before the stage game is re-solved; 1 re-solves every
    iteration.

    Returns (q_red, q_blue, strategy_red, strategy_blue).
    """
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}, choose from {list(SOLVERS)}")
    if resync_every < 1:
        raise ValueError("resync_every must be >= 1")
    solve_stage = SOLVERS[solver]

    q_red = payoff_red.copy()
    q_blue = payoff_blue.copy()
    strategy_red = strategy_blue = None

    for i in range(max_outer_iter):
        if i % resync_every == 0 or strategy_red is None:
            strategy_red, strategy_blue = solve_stage(
                q_red, q_blue, fictitious_play_iterations, seed
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
