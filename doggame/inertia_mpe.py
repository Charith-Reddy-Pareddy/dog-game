"""Is holding the one-shot Nash actions a Markov-perfect (feedback) equilibrium of the game with dog inertia?

With inertia the state is the dog's position x, and dog' = w0 * x + (1 - w0) * (w * a_red + (1 - w) * a_blue).
If both players always play their one-shot Nash actions, the dog moves geometrically toward d* (the one-shot Nash
dog), so each player's value has a closed form:

    V(x) = -[ ||d* - h||^2 / (1 - g) + 2 w0 / (1 - g w0) (d* - h) . (x - d*) + w0^2 / (1 - g w0^2) ||x - d*||^2 ]

with discount g. By the one-shot deviation principle (discounted rewards, bounded state), the profile is a Markov-perfect
equilibrium exactly when no player can gain by changing its action for one round, at any state, and then returning to the
profile. That one-round problem is a concave quadratic in each coordinate of the next dog position, over the interval the
player can reach, so its best answer and gain are exact.
"""

import numpy as np


def value(x, d_star, house, w0, discount):
    """Discounted reward from state x when both players hold the actions that put the dog at d_star."""
    e, gap = np.asarray(x, dtype=float) - d_star, d_star - np.asarray(house, dtype=float)
    return -(gap @ gap / (1 - discount) + 2 * w0 / (1 - discount * w0) * gap @ e + w0 ** 2 / (1 - discount * w0 ** 2) * e @ e)


def one_shot_gain(x, house, own, other, own_weight, w0, discount):
    """The most a player gains at state x by changing its action for one round. Returns (gain, best action)."""
    x, house, own, other = (np.asarray(v, dtype=float) for v in (x, house, own, other))
    d_star = own_weight * own + (1 - own_weight) * other
    slope = w0 / (1 - discount * w0) * (d_star - house)
    curve = w0 ** 2 / (1 - discount * w0 ** 2)
    score = lambda z: -(z - house) ** 2 - discount * (2 * slope * (z - d_star) + curve * (z - d_star) ** 2)  # per coordinate
    low = w0 * x + (1 - w0) * (1 - own_weight) * other     # next dog if the player plays 0 in a coordinate
    high = low + (1 - w0) * own_weight                       # ... and if it plays 1
    target = (house - discount * slope + discount * curve * d_star) / (1 + discount * curve)
    best = np.clip(target, low, high)
    held = w0 * x + (1 - w0) * d_star
    gain = float(np.sum(score(best) - score(held)))
    action = (best - low) / ((1 - w0) * own_weight)
    return max(gain, 0.0), action


def check_mpe(houses, w, w0, actions, discount=0.9, grid=41):
    """For each player, the largest one-round gain over a grid of dog positions, where it happens, and the gain at the
    starting state (the dog between the two houses)."""
    house_red, house_blue = (np.asarray(h, dtype=float) for h in houses)
    action_red, action_blue = (np.asarray(a, dtype=float) for a in actions)
    start = w * house_red + (1 - w) * house_blue
    axis = np.linspace(0.0, 1.0, grid)
    result = {}
    for name, house, own, other, weight in (("red", house_red, action_red, action_blue, w), ("blue", house_blue, action_blue, action_red, 1 - w)):
        gains = [(one_shot_gain((x, y), house, own, other, weight, w0, discount)[0], (x, y)) for x in axis for y in axis]
        worst, where = max(gains)
        result[name] = dict(max_gain=worst, worst_state=list(where), start_gain=one_shot_gain(start, house, own, other, weight, w0, discount)[0])
    return result
