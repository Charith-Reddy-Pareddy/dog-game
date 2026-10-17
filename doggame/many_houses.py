"""The dog game with many houses, one coordinate at a time.

N players with houses h_i in [0, 1] and weights w_i summing to 1 put the dog at d = sum_i w_i a_i. The one-shot game
is still an exact potential game, and its equilibrium dog is the weighted generalized median found by Renault and
Trannoy (2005) for the average voting rule: with S(d) the total weight of houses above d, players above d* push to 1,
players below push to 0, and d* is the unique point with S(d*) <= d* <= S(d*) + (weight of houses at d*). At most one
player, the pivot, ends strictly inside the box (its house is d*).

With dog inertia, the one-round argument of Proposition 1 (docs/INERTIA_PROOFS.md) never used the number of players,
so it holds verbatim: holding the one-shot actions is Markov-perfect iff no pivot is strictly inside (for w0 > 0) and
every player at an edge has slack at least rho(w0, g). Many houses crowd d*, so the smallest slack, and with it the
largest inertia that keeps the one-shot profile Markov-perfect, shrinks as N grows.
"""

import numpy as np


def rho(w0, discount):
    return w0 * (1 - discount * w0) / (1 - discount * w0 ** 2)


def one_shot(houses, weights):
    """The one-shot equilibrium of one coordinate: (actions, d*, index of the pivot or None)."""
    houses, weights = np.asarray(houses, dtype=float), np.asarray(weights, dtype=float)
    above = 0.0
    for i in np.argsort(-houses):  # walk down from the highest house
        if houses[i] <= above:      # the dog stops above this house: everyone left pushes down
            break
        if houses[i] <= above + weights[i]:  # this player stops the dog on its own house
            actions = (houses > houses[i]).astype(float)
            actions[i] = (houses[i] - above) / weights[i]
            return actions, float(houses[i]), int(i)
        above += weights[i]
    return (houses > above).astype(float), float(above), None


def slacks(houses, actions, d_star):
    """Each edge player's slack: (h - d*) / (1 - d*) at the top, (d* - h) / d* at the bottom (nan for the pivot)."""
    houses = np.asarray(houses, dtype=float)
    top = (houses - d_star) / (1 - d_star) if d_star < 1 else np.zeros_like(houses)
    bottom = (d_star - houses) / d_star if d_star > 0 else np.zeros_like(houses)
    return np.where(actions == 1, top, np.where(actions == 0, bottom, np.nan))


def critical_inertia(houses, weights, discount=0.9, tol=1e-12):
    """The largest w0 at which holding the one-shot actions is Markov-perfect (0 when a pivot is inside the box)."""
    actions, d_star, pivot = one_shot(houses, weights)
    if pivot is not None:
        return 0.0
    s = float(np.min(slacks(houses, actions, d_star)))
    low, high = 0.0, 1.0  # rho rises from 0 to 1 on [0, 1)
    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if rho(mid, discount) <= s else (low, mid)
    return low


def mean_field_dog(cdf, tol=1e-12):
    """The limit of d* with equal weights and houses drawn from `cdf`: the d with d = 1 - cdf(d)."""
    low, high = 0.0, 1.0
    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if mid < 1 - cdf(mid) else (low, mid)
    return (low + high) / 2


def next_dog(x, targets, weights, w0):
    """The next dog position (vectorized over x) when every player aims it at its own target: the round is the one-shot
    game with the targets as houses, so in action units the dog lands at the generalized median
    clip(max_k min(W_k, u_(k)), 0, 1), with u the shifted targets (t - w0 x) / (1 - w0) sorted downward and W their
    cumulative weights."""
    order = np.argsort(-np.asarray(targets, dtype=float))
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    shifted = (np.asarray(targets, dtype=float)[order][None, :] - w0 * np.asarray(x, dtype=float)[:, None]) / (1 - w0)
    return w0 * np.asarray(x, dtype=float) + (1 - w0) * np.clip(np.max(np.minimum(cumulative[None, :], shifted), 1), 0, 1)


def _values(x, targets, houses, weights, w0, discount, rounds):
    """Each player's discounted payoff from dog positions x when everyone aims at the targets."""
    x, total = np.asarray(x, dtype=float), np.zeros((len(houses), len(x)))
    for k in range(rounds):
        x = next_dog(x, targets, weights, w0)
        total -= discount ** k * (x[None, :] - np.asarray(houses, dtype=float)[:, None]) ** 2
    return total


def target_equilibrium(houses, weights, w0, discount=0.9, grid=801, rounds=200, tol=1e-9, max_iter=200):
    """Targets at which aiming the dog is a feedback equilibrium of the many-house game with inertia, by damped
    fixed-point iteration: each target is where its player's one-round objective peaks under the current targets.
    Starts from the two-player formula d* + k (h - d*). Returns (targets, iterations)."""
    houses = np.asarray(houses, dtype=float)
    _, d_star, _ = one_shot(houses, weights)
    targets = d_star + (1 - discount * w0 ** 2) / (1 - discount * w0) * (houses - d_star)
    zs = np.linspace(0, 1, grid)
    for it in range(1, max_iter + 1):
        objective = -(zs[None, :] - houses[:, None]) ** 2 + discount * _values(zs, targets, houses, weights, w0, discount, rounds)
        new = zs[objective.argmax(1)]
        if np.abs(new - targets).max() < tol:
            return new, it
        targets = 0.5 * targets + 0.5 * new
    return targets, max_iter


def deviation_gain(targets, houses, weights, w0, discount=0.9, states=21, grid=201, rounds=200):
    """The most any player gains at any of `states` dog positions by changing its action for one round and then
    aiming at the targets again (others aiming throughout)."""
    houses, weights, best = np.asarray(houses, dtype=float), np.asarray(weights, dtype=float), 0.0
    for x in np.linspace(0, 1, states):
        actions = one_shot((np.asarray(targets) - w0 * x) / (1 - w0), weights)[0]
        for i in range(len(houses)):
            own = np.append(np.linspace(0, 1, grid), actions[i])
            z = w0 * x + (1 - w0) * (actions @ weights - weights[i] * actions[i] + weights[i] * own)
            total = -(z - houses[i]) ** 2 + discount * _values(z, targets, houses, weights, w0, discount, rounds)[i]
            best = max(best, float(total[:-1].max() - total[-1]))
    return best


def objective_peaks(targets, houses, weights, w0, discount=0.9, grid=4001, rounds=200):
    """For each player, the number of local maxima of its one-round objective under the targets; aiming at a target
    is a best reply wherever that objective is single-peaked."""
    zs, houses = np.linspace(0, 1, grid), np.asarray(houses, dtype=float)
    objective = -(zs[None, :] - houses[:, None]) ** 2 + discount * _values(zs, targets, houses, weights, w0, discount, rounds)
    return [int(np.sum(np.diff(np.sign(np.diff(row))) < 0)) for row in objective]
