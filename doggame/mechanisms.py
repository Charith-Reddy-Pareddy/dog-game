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

``rule`` can also be a tuple of N - 1 phantoms: the generalized median of the reports and the phantoms
(Moulin 1980), the general anonymous strategy-proof rule in one shot. Phantoms (0, ..., 0, 1, ..., 1)
give the median rule; with equal weights, phantoms (1/N, ..., (N-1)/N) give the same map from targets
to next outcome as the mean rule (the max-min form of next_dog), so the same target-form equilibria.

With ``shock`` (Gauss-Hermite nodes and weights, from ``shocks``) a normal shock is added to each next
outcome, which is then clipped to [0, 1]. A player's one-round objective still depends only on the
outcome before the shock, so target form survives. Values then come from the stationary Bellman
equation on a grid of outcomes (interpolated), not from simulated paths: fixed draws per round would
make the problem non-stationary, and a one-round deviation could then exploit the particular draws.
"""

import numpy as np

from doggame.many_houses import next_dog as next_mean
from doggame.many_houses import one_shot

RULES = ("mean", "median")


def next_phantom(x, targets, phantoms, w0):
    """The next outcome under the generalized median with ``phantoms`` when every player aims at its target."""
    x = np.asarray(x, dtype=float)
    shifted = np.clip((np.asarray(targets, dtype=float)[None, :] - w0 * x[:, None]) / (1 - w0), 0, 1)
    both = np.concatenate([shifted, np.broadcast_to(np.asarray(phantoms, dtype=float), (len(x), len(phantoms)))], 1)
    return w0 * x + (1 - w0) * np.median(both, axis=1)


def next_median(x, targets, w0):
    """The next outcome under the median rule when every player aims at its target (vectorized over x)."""
    x = np.asarray(x, dtype=float)
    return np.clip(np.median(targets), w0 * x, w0 * x + 1 - w0)


def compensated(x, reports, phantoms, w0):
    """The inertia-compensated receiver: the next outcome is as close to the generalized median of the reports
    and ``phantoms`` as the inertia allows, instead of w0 x + (1 - w0) times that median."""
    r = np.asarray(reports, dtype=float)
    m = np.median(np.concatenate([r, np.broadcast_to(np.asarray(phantoms, dtype=float), r.shape[:-1] + (len(phantoms),))], -1), axis=-1)
    return np.clip(m, w0 * np.asarray(x, dtype=float), w0 * np.asarray(x, dtype=float) + 1 - w0)


def step(rule, x, targets, weights, w0):
    if isinstance(rule, tuple):
        return next_phantom(x, targets, rule, w0)
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


def target_equilibrium(rule, houses, weights, w0, discount=0.9, grid=801, rounds=200, tol=1e-9, max_iter=300, shock=None):
    """Targets in equilibrium for the rule, by damped fixed-point iteration from the houses."""
    houses = np.asarray(houses, dtype=float)
    targets, zs = houses.copy(), np.linspace(0, 1, grid)
    for it in range(1, max_iter + 1):
        if shock is None:
            objective = -(zs[None, :] - houses[:, None]) ** 2 + discount * _values(rule, zs, targets, houses, weights, w0, discount, rounds)
        else:
            objective = -_shock_losses(lambda x: step(rule, x, targets, weights, w0), zs, houses, discount, shock)
        new = zs[objective.argmax(1)]
        if np.abs(new - targets).max() < tol:
            return new, it
        targets = 0.5 * targets + 0.5 * new
    return targets, max_iter


def shocks(noise, w0, filtered=True, nodes=9):
    """Nodes and weights for the shock: normal with standard deviation noise, times (1 - w0) when the
    shock comes with the new aggregate (filtered by the inertia) rather than hitting the outcome itself."""
    x, p = np.polynomial.hermite_e.hermegauss(nodes)
    return noise * (1 - w0 if filtered else 1) * x, p / p.sum()


def _shock_losses(move, z, houses, discount, shock, grid=401, sweeps=200):
    """Each player's expected discounted loss when the next outcome before its shock is z and the
    outcome then follows ``move`` and the shocks."""
    nodes, probs = shock
    houses, ys = np.asarray(houses, dtype=float)[:, None, None], np.linspace(0, 1, grid)
    land = lambda v: np.clip(np.asarray(v, dtype=float)[:, None] + nodes, 0, 1)  # noqa: E731
    y, total = land(move(ys)), np.zeros((len(houses), grid))
    for _ in range(sweeps):  # from each realized outcome on the grid
        total = ((y[None] - houses) ** 2 + discount * np.stack([np.interp(y, ys, t) for t in total])) @ probs
    y = land(z)
    return ((y[None] - houses) ** 2 + discount * np.stack([np.interp(y, ys, t) for t in total])) @ probs


def deviation_gain(rule, policy, houses, weights, w0, discount=0.9, states=21, grid=201, rounds=200, compensate=False):
    """The most any player gains, at any of ``states`` outcomes, by changing its report for one round and
    then following ``policy`` again; ``policy(x)`` gives everyone's reports at outcome x. With ``compensate``
    the receiver is the inertia-compensated one. ``rule`` can also be any function of the reports (last axis)."""
    houses, weights, best = np.asarray(houses, dtype=float), np.asarray(weights, dtype=float), 0.0
    if callable(rule):
        agg = rule
    elif isinstance(rule, tuple):
        agg = lambda r: np.median(np.concatenate([r, np.broadcast_to(rule, r.shape[:-1] + (len(rule),))], -1), axis=-1)  # noqa: E731
    else:
        agg = (lambda r: r @ weights) if rule == "mean" else (lambda r: np.median(r, axis=-1))
    move = (lambda x, r: np.clip(agg(r), w0 * x, w0 * x + 1 - w0)) if compensate else (lambda x, r: w0 * x + (1 - w0) * agg(r))  # noqa: E731
    for x in np.linspace(0, 1, states):
        now = policy(x)
        for i in range(len(houses)):
            trial = np.tile(now, (grid + 1, 1))
            trial[:-1, i] = np.linspace(0, 1, grid)  # every deviation, and the policy's own report last
            z = move(x, trial)
            total = -(z - houses[i]) ** 2
            for k in range(1, rounds):
                z = move(z, np.array([policy(v) for v in z]))
                total -= discount ** k * (z - houses[i]) ** 2
            best = max(best, float(total[:-1].max() - total[-1]))
    return best


def limit_coefficients(n):
    """c_k = E[(x - mean house) 1{k houses above x}] for n houses and a start x, all uniform on [0, 1]:
    (n - 2k) / (2n(n + 2)), k = 0..n. As inertia tends to 1 a generalized median receiver's loss is, to first
    order, a positive multiple of sum_k c_k alpha_k (Proposition 8)."""
    return np.array([(n - 2 * k) / (2 * n * (n + 2)) for k in range(n + 1)])


def order_statistic_phantoms(n, j):
    """Phantoms of the receiver that moves toward the j-th smallest target: (n - j) phantoms at 0, j - 1 at 1."""
    return tuple([0.0] * (n - j) + [1.0] * (j - 1))


def best_order_statistic(samples):
    """For target draws ``samples`` (draws x n) and a uniform start, the order statistic j (1-based) of the slow
    receiver with the lowest first-order loss: the one minimizing E[(h_(j) - mean target)^2] (Proposition 8)."""
    ordered, mean = np.sort(samples, axis=1), samples.mean(axis=1, keepdims=True)
    return int(np.argmin(((ordered - mean) ** 2).mean(axis=0))) + 1


def compensated_criterion(samples, phantoms):
    """The first-order term of a slow compensated receiver, up to the factor 2N (1 - w0) / (1 - g)^2:
    E[s^2 / 2 - s hbar], with s the generalized median of the targets and phantoms. It ranks receivers as
    E[(s - hbar)^2] does, the static criterion."""
    s = np.median(np.concatenate([samples, np.broadcast_to(np.asarray(phantoms, dtype=float), (len(samples), len(phantoms)))], 1), axis=1)
    return float(np.mean(s ** 2 / 2 - s * samples.mean(1)))


def planner_step(x, houses, w0):
    """The planner moves the outcome as close to the mean house as it can this round (optimal without shocks)."""
    x = np.asarray(x, dtype=float)
    return np.clip(np.mean(houses), w0 * x, w0 * x + 1 - w0)


def planner_target(houses, w0, discount=0.9, shock=None, grid=401):
    """The planner's target with shocks: the summed loss is n (z - mean house)^2 plus a constant, so the
    planner is a single player whose house is the mean house; with clipped shocks its target can differ
    from that house. Its next outcome is next_median(x, [target], w0)."""
    return float(target_equilibrium("median", [np.mean(houses)], [1.0], w0, discount, grid=grid, shock=shock)[0][0])


def social_loss(next_fn, houses, discount=0.9, starts=101, rounds=200, shock=None):
    """Discounted summed loss of all players, averaged over evenly spread starting outcomes."""
    x, total = np.linspace(0, 1, starts), np.zeros(starts)
    houses = np.asarray(houses, dtype=float)
    if shock is not None:
        return float(_shock_losses(next_fn, next_fn(x), houses, discount, shock).sum(0).mean())
    for k in range(rounds):
        x = next_fn(x)
        total += discount ** k * ((x[:, None] - houses[None, :]) ** 2).sum(1)
    return float(total.mean())


def compensated_losses(houses, shapes, w0, discount=0.9, starts=101, rounds=200):
    """social_loss of the compensated receiver under truthful reports, for many phantom shapes at once: the
    outcome moves as close as it can to s = the generalized median of the houses and the shape's phantoms."""
    houses = np.asarray(houses, dtype=float)
    s = np.array([np.median(np.concatenate([houses, np.asarray(ph, dtype=float)])) for ph in shapes])[:, None]
    x, total = np.tile(np.linspace(0, 1, starts), (len(s), 1)), np.zeros((len(s), starts))
    for k in range(rounds):
        x = np.clip(s, w0 * x, w0 * x + 1 - w0)
        total += discount ** k * ((x[..., None] - houses) ** 2).sum(-1)
    return total.mean(1)


def steady_state(next_fn, start=0.5, rounds=2000):
    x = np.array([start])
    for _ in range(rounds):
        x = next_fn(x)
    return float(x[0])
