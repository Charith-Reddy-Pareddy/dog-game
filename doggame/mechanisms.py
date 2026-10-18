"""Mean against median aggregation when the outcome has inertia (one coordinate).

N players with houses h_i report points r_i in [0, 1]; the outcome moves to
z = w0 x + (1 - w0) * rule(r), where ``rule`` is the weighted mean (the dog game, whose one-shot
equilibrium is the generalized median of Renault and Trannoy 2005) or the median of the reports
(strategy-proof in one shot, Moulin 1980). Each player loses (z - h_i)^2 per round, discounted.

In both rules a player's one-round objective depends only on the next outcome, so equilibria come in
target form: each player reports so as to aim the next outcome at its own target. Given targets,
the next outcome is, for the mean rule, the generalized median of the targets
(doggame.many_houses.next_dog), and for the median rule simply the median target clipped to what can
be reached this round. The targets solve a fixed point, found by damped iteration. A planner who
minimizes the summed loss moves the outcome as close to the mean house as it can each round.
"""

import numpy as np

from doggame.many_houses import next_dog as next_mean
from doggame.many_houses import one_shot

RULES = ("mean", "median")


def next_median(x, targets, w0):
    """The next outcome under the median rule when every player aims at its target (vectorized over x)."""
    x = np.asarray(x, dtype=float)
    return np.clip(np.median(targets), w0 * x, w0 * x + 1 - w0)


def step(rule, x, targets, weights, w0):
    return next_mean(x, targets, weights, w0) if rule == "mean" else next_median(x, targets, w0)


def reports(rule, x, targets, weights, w0):
    """Each player's report at outcome x when all aim at their targets."""
    shifted = (np.asarray(targets, dtype=float) - w0 * x) / (1 - w0)
    if rule == "mean":
        return one_shot(shifted, weights)[0]
    return np.clip(shifted, 0, 1)


def _values(rule, x, targets, houses, weights, w0, discount, rounds):
    """Each player's discounted loss (negative) from outcomes x when everyone aims at the targets."""
    x, total = np.asarray(x, dtype=float), np.zeros((len(houses), len(x)))
    for k in range(rounds):
        x = step(rule, x, targets, weights, w0)
        total -= discount ** k * (x[None, :] - np.asarray(houses, dtype=float)[:, None]) ** 2
    return total


def target_equilibrium(rule, houses, weights, w0, discount=0.9, grid=801, rounds=200, tol=1e-9, max_iter=300):
    """Targets in equilibrium for the rule, by damped fixed-point iteration from the houses."""
    houses = np.asarray(houses, dtype=float)
    targets, zs = houses.copy(), np.linspace(0, 1, grid)
    for it in range(1, max_iter + 1):
        objective = -(zs[None, :] - houses[:, None]) ** 2 + discount * _values(rule, zs, targets, houses, weights, w0, discount, rounds)
        new = zs[objective.argmax(1)]
        if np.abs(new - targets).max() < tol:
            return new, it
        targets = 0.5 * targets + 0.5 * new
    return targets, max_iter


def deviation_gain(rule, policy, houses, weights, w0, discount=0.9, states=21, grid=201, rounds=200):
    """The most any player gains, at any of ``states`` outcomes, by changing its report for one round and
    then following ``policy`` again; ``policy(x)`` gives everyone's reports at outcome x."""
    houses, weights, best = np.asarray(houses, dtype=float), np.asarray(weights, dtype=float), 0.0
    agg = (lambda r: r @ weights) if rule == "mean" else (lambda r: np.median(r, axis=-1))
    for x in np.linspace(0, 1, states):
        now = policy(x)
        for i in range(len(houses)):
            trial = np.tile(now, (grid + 1, 1))
            trial[:-1, i] = np.linspace(0, 1, grid)  # every deviation, and the policy's own report last
            z = w0 * x + (1 - w0) * agg(trial)
            total = -(z - houses[i]) ** 2
            for k in range(1, rounds):
                z = w0 * z + (1 - w0) * np.array([agg(policy(v)) for v in z])
                total -= discount ** k * (z - houses[i]) ** 2
            best = max(best, float(total[:-1].max() - total[-1]))
    return best


def planner_step(x, houses, w0):
    """The planner moves the outcome as close to the mean house as it can this round."""
    x = np.asarray(x, dtype=float)
    return np.clip(np.mean(houses), w0 * x, w0 * x + 1 - w0)


def social_loss(next_fn, houses, discount=0.9, starts=101, rounds=200):
    """Discounted summed loss of all players, averaged over evenly spread starting outcomes."""
    x, total = np.linspace(0, 1, starts), np.zeros(starts)
    houses = np.asarray(houses, dtype=float)
    for k in range(rounds):
        x = next_fn(x)
        total += discount ** k * ((x[:, None] - houses[None, :]) ** 2).sum(1)
    return float(total.mean())


def steady_state(next_fn, start=0.5, rounds=2000):
    x = np.array([start])
    for _ in range(rounds):
        x = next_fn(x)
    return float(x[0])
