"""Exact (not approximate) Nash equilibrium solvers for bimatrix stage
games, as an alternative to fictitious play's best-response dynamics.

Fictitious play (doggame.fictitious_play) only approximates a stage
game's Nash equilibrium, and leaves a small numerical residual from its
random initial pick (see doggame.verify's `tol` discussion). The
project's planning notes call out two exact alternatives instead:

- For a general-sum bimatrix game (what the dog game's discretized
  stage game actually is): NashPy's Lemke-Howson algorithm, which
  pivots to an exact equilibrium in polynomial practice, unlike support
  enumeration's worst-case-exponential search over every support pair.
- For a zero-sum game specifically (what the *soccer* game's stage
  game is -- the dog game is general-sum, so this path is unused by
  the rest of this package, but is included because it is the correct
  exact method for that case, and is reused when this package's
  machinery is eventually pointed at soccer): an LP via scipy, using
  the standard minimax-as-linear-program reduction. A general-sum game
  cannot be solved this way -- LP duality only holds for zero-sum
  games -- so this is not a drop-in replacement for
  `lemke_howson_nash`.
"""

import numpy as np
from scipy.optimize import linprog


def lemke_howson_nash(payoff_red, payoff_blue, initial_dropped_label=0, epsilon=0.0, seed=0):
    """An exact Nash equilibrium of a general-sum bimatrix game via
    NashPy's Lemke-Howson algorithm.

    Returns (strategy_red, strategy_blue). A bimatrix game can have
    multiple equilibria; Lemke-Howson returns whichever one the pivot
    path starting from `initial_dropped_label` finds, not all of them.

    `epsilon` matches the planning notes' "[payoff matrix] + epsilon"
    shorthand for breaking exact payoff ties before solving: a tied
    (degenerate) game can have infinitely many equilibria sharing the
    same support, which is a harder case for pivoting methods in
    general, even though NashPy's implementation has not been observed
    to fail on the degenerate cases this project actually exercises.
    `epsilon = 0` (the default) solves the payoff matrices exactly as
    given; `epsilon > 0` adds a small uniform-random perturbation
    (seeded, so it is reproducible) to both matrices first.
    """
    import nashpy as nash

    payoff_red = np.asarray(payoff_red, dtype=float)
    payoff_blue = np.asarray(payoff_blue, dtype=float)
    if epsilon:
        rng = np.random.default_rng(seed)
        payoff_red = payoff_red + rng.uniform(-epsilon, epsilon, size=payoff_red.shape)
        payoff_blue = payoff_blue + rng.uniform(-epsilon, epsilon, size=payoff_blue.shape)

    game = nash.Game(payoff_red, payoff_blue)
    strategy_red, strategy_blue = game.lemke_howson(initial_dropped_label=initial_dropped_label)
    return np.asarray(strategy_red, dtype=float), np.asarray(strategy_blue, dtype=float)


def zero_sum_lp_nash(payoff_red):
    """An exact Nash equilibrium (and game value) of a zero-sum
    bimatrix game, via the standard minimax-as-linear-program
    reduction (row player maximizes their guaranteed value; column
    player's payoff is `-payoff_red`).

    Returns (strategy_red, strategy_blue, value), the game's value
    from the row player's (red's) perspective.
    """
    payoff_red = np.asarray(payoff_red, dtype=float)
    n_red, n_blue = payoff_red.shape

    # Row player (red): maximize v subject to
    #   sum_i strategy_red[i] * payoff_red[i, j] >= v  for every column j
    #   sum_i strategy_red[i] == 1, strategy_red >= 0
    # linprog minimizes, so maximize v by minimizing -v; variables are
    # [strategy_red (n_red), v].
    c = np.zeros(n_red + 1)
    c[-1] = -1.0

    # -sum_i strategy_red[i] * payoff_red[i, j] + v <= 0  for every j
    A_ub = np.hstack([-payoff_red.T, np.ones((n_blue, 1))])
    b_ub = np.zeros(n_blue)

    A_eq = np.zeros((1, n_red + 1))
    A_eq[0, :n_red] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_red + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"zero-sum LP failed to solve: {result.message}")
    strategy_red = result.x[:n_red]
    value = result.x[-1]

    # Column player (blue): the dual, solved the same way on -payoff_red.T
    payoff_blue = -payoff_red.T
    c2 = np.zeros(n_blue + 1)
    c2[-1] = -1.0
    A_ub2 = np.hstack([-payoff_blue.T, np.ones((n_red, 1))])
    b_ub2 = np.zeros(n_red)
    A_eq2 = np.zeros((1, n_blue + 1))
    A_eq2[0, :n_blue] = 1.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0.0, None)] * n_blue + [(None, None)]
    result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not result2.success:
        raise RuntimeError(f"zero-sum LP failed to solve: {result2.message}")
    strategy_blue = result2.x[:n_blue]

    return strategy_red, strategy_blue, value
