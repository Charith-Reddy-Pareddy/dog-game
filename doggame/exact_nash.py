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

import warnings
from itertools import cycle

import numpy as np
from nashpy.linalg import create_col_tableau, create_row_tableau
from scipy.optimize import linprog


class _TooManyPivots(Exception):
    pass


def _capped_lemke_howson(payoff_red, payoff_blue, label, max_pivots):
    """NashPy's Lemke-Howson loop, but giving up after `max_pivots` pivots.

    On some games the pivoting cycles forever. Finished games need only a
    few dozen pivots (at most 22 on random 10x10 games), so a cap is a safe
    and, unlike a time limit, perfectly repeatable way to stop a cycle.
    """
    col_tableau = create_col_tableau(payoff_red, True)
    row_tableau = create_row_tableau(payoff_blue, True)
    if label in row_tableau.non_basic_variables:
        tableaux = cycle((row_tableau, col_tableau))
    else:
        tableaux = cycle((col_tableau, row_tableau))

    entering = label
    for _ in range(max_pivots):
        entering = next(tableaux).pivot_and_drop_label(entering)
        labels = col_tableau.non_basic_variables.union(row_tableau.non_basic_variables)
        if labels == col_tableau.labels:
            strategy_red = row_tableau.to_strategy(col_tableau.non_basic_variables)
            strategy_blue = col_tableau.to_strategy(row_tableau.non_basic_variables)
            return np.asarray(strategy_red, dtype=float), np.asarray(strategy_blue, dtype=float)
    raise _TooManyPivots()


def _is_equilibrium(strategy_red, strategy_blue, payoff_red, payoff_blue, tol=1e-6):
    """True if both strategies are valid and neither player gains from a
    one-sided switch to any single action."""
    shapes_ok = strategy_red.shape == (payoff_red.shape[0],) and strategy_blue.shape == (payoff_red.shape[1],)
    if not shapes_ok or not (np.isfinite(strategy_red).all() and np.isfinite(strategy_blue).all()):
        return False
    if abs(strategy_red.sum() - 1) > tol or abs(strategy_blue.sum() - 1) > tol:
        return False
    value_red = strategy_red @ payoff_red @ strategy_blue
    value_blue = strategy_red @ payoff_blue @ strategy_blue
    return (
        (payoff_red @ strategy_blue).max() <= value_red + tol
        and (strategy_red @ payoff_blue).max() <= value_blue + tol
    )


def lemke_howson_nash(payoff_red, payoff_blue, initial_dropped_label=0, epsilon=0.0, seed=0, max_pivots=None):
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

    NashPy's pivoting can overflow or cycle forever on some games, so
    each answer is checked to be a real equilibrium. If it isn't (or the
    attempt needs more than `max_pivots` pivots, by default ten times the
    number of actions), the next starting label is tried, and a
    RuntimeError is raised if none of them work. The result depends only on
    the payoffs, never on how fast the machine is.
    """
    payoff_red = np.asarray(payoff_red, dtype=float)
    payoff_blue = np.asarray(payoff_blue, dtype=float)
    if epsilon:
        rng = np.random.default_rng(seed)
        payoff_red = payoff_red + rng.uniform(-epsilon, epsilon, size=payoff_red.shape)
        payoff_blue = payoff_blue + rng.uniform(-epsilon, epsilon, size=payoff_blue.shape)

    n_labels = sum(payoff_red.shape)
    max_pivots = max_pivots or 10 * n_labels
    for offset in range(n_labels):
        label = (initial_dropped_label + offset) % n_labels
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # overflow noise from failed pivots
                strategy_red, strategy_blue = _capped_lemke_howson(payoff_red, payoff_blue, label, max_pivots)
        except _TooManyPivots:
            continue
        if _is_equilibrium(strategy_red, strategy_blue, payoff_red, payoff_blue):
            return strategy_red, strategy_blue
    raise RuntimeError("Lemke-Howson found no valid equilibrium from any starting label")


def _maximin_strategy(payoff):
    """The row player's maximin mixed strategy and value for a payoff
    matrix, solved as a linear program over [strategy, value]."""
    n_rows, n_cols = payoff.shape
    cost = np.zeros(n_rows + 1)
    cost[-1] = -1.0  # linprog minimizes, so minimize -value

    # for every column j: value - sum_i strategy[i] * payoff[i, j] <= 0
    A_ub = np.hstack([-payoff.T, np.ones((n_cols, 1))])
    A_eq = np.append(np.ones(n_rows), 0.0).reshape(1, -1)
    bounds = [(0.0, None)] * n_rows + [(None, None)]

    result = linprog(
        cost, A_ub=A_ub, b_ub=np.zeros(n_cols), A_eq=A_eq, b_eq=[1.0],
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"zero-sum LP failed to solve: {result.message}")
    return result.x[:n_rows], result.x[-1]


def zero_sum_lp_nash(payoff_red):
    """An exact Nash equilibrium of a zero-sum game via the standard
    minimax linear program. Blue's payoff is `-payoff_red`.

    Returns (strategy_red, strategy_blue, value), with the value from
    red's point of view.
    """
    payoff_red = np.asarray(payoff_red, dtype=float)
    strategy_red, value = _maximin_strategy(payoff_red)
    strategy_blue, _ = _maximin_strategy(-payoff_red.T)
    return strategy_red, strategy_blue, value
