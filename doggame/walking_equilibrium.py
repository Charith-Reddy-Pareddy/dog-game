"""Is walking to the one-shot Nash corners an equilibrium of the walking game?

Both players follow a fixed path: walk straight to your one-shot corner at full
speed, then stay. If one player follows its path, the other faces a known,
deterministic opponent, so its best response is an optimal-control problem:
choose positions p_1..p_T, each within `max_step` of the last and inside the
square, to maximize the discounted sum of its rewards. The profile is a Nash
equilibrium when neither player's best response beats following its own path.
(With no dog inertia the reward each step depends only on the two current
positions, so no memory of the dog's position enters.)

This checks the path profile against all deviations from one player. It is not a
solution of the full dynamic game, whose Markov-perfect equilibrium is not computed.
Starting both players on the corners checks the resting state itself.

Why walking straight to the corner is not exactly a best response: each step's best
position is the unconstrained optimum, which lies outside the square, so the best
path heads toward that point and not toward the corner. The gains are small.
"""

import numpy as np
from scipy.optimize import minimize


def corner_path(start, corner, max_step, steps):
    """Positions at times 1..steps of walking straight to `corner` at full speed, then staying."""
    start, corner = np.asarray(start, dtype=float), np.asarray(corner, dtype=float)
    distance = np.linalg.norm(corner - start)
    direction = (corner - start) / distance if distance > 0 else np.zeros(2)
    return np.array([start + direction * min(max_step * (t + 1), distance) for t in range(steps)])


def discounted_reward(path, opponent_path, house, w, own_weight, discount):
    """Sum over steps of discount**t times minus the squared distance from the dog to `house`.
    `own_weight` is the player's weight in the dog's position (w for red, 1 - w for blue)."""
    dog = own_weight * path + (1 - own_weight) * opponent_path
    return float(-sum(discount ** t * np.sum((dog[t] - house) ** 2) for t in range(len(path))))


def best_response(start, opponent_path, house, w, own_weight, max_step, discount, guess):
    """The player's best discounted reward against a fixed opponent path, and the path that achieves it."""
    steps = len(opponent_path)
    start = np.asarray(start, dtype=float)

    def unpack(x):
        return x.reshape(steps, 2)

    def objective(x):
        return -discounted_reward(unpack(x), opponent_path, house, w, own_weight, discount)

    def speed(x):  # at least 0 when each move is within the maximum step
        path = np.vstack([start, unpack(x)])
        return max_step ** 2 - np.sum(np.diff(path, axis=0) ** 2, axis=1)

    result = minimize(objective, np.asarray(guess, dtype=float).ravel(), method="SLSQP", bounds=[(0.0, 1.0)] * (2 * steps),
                      constraints=[{"type": "ineq", "fun": speed}], options={"maxiter": 500, "ftol": 1e-12})
    return -result.fun, unpack(result.x)


def check_profile(houses, w, corners, max_step=0.2, steps=15, discount=0.9, starts=None):
    """For each player, the reward of following the corner path, the best response's reward, and the gain from deviating.
    `houses` and `corners` are ((red x, y), (blue x, y)) each. Players start on their houses unless `starts` is given."""
    (house_red, house_blue), (corner_red, corner_blue) = houses, corners
    start_red, start_blue = (np.clip(house_red, 0, 1), np.clip(house_blue, 0, 1)) if starts is None else map(np.asarray, starts)
    path_red = corner_path(start_red, corner_red, max_step, steps)
    path_blue = corner_path(start_blue, corner_blue, max_step, steps)
    result = {}
    for name, start, house, own, other, weight in (("red", start_red, house_red, path_red, path_blue, w), ("blue", start_blue, house_blue, path_blue, path_red, 1 - w)):
        followed = discounted_reward(own, other, np.asarray(house), w, weight, discount)
        best, _ = best_response(start, other, np.asarray(house), w, weight, max_step, discount, guess=own)
        result[name] = dict(followed=followed, best=max(best, followed), gain=max(best - followed, 0.0))
    return result
