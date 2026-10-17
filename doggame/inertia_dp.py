"""The exact feedback (Markov-perfect) equilibrium of the dog game with inertia, by dynamic programming.

Rewards are sums over the two coordinates, and both the dog's motion and the action boxes act coordinate by
coordinate, so the 2D game splits into two independent 1D games. A pair of 1D equilibria is an equilibrium of the
2D game (a player's change in one coordinate never affects the other). Each 1D game has state z in [0, 1] (one
coordinate of the dog), actions in [0, 1] and z' = w0 z + (1 - w0) (w a_red + (1 - w) a_blue); it is solved by Nash
value iteration on a grid of states and actions, with the continuation values interpolated linearly.
"""

import numpy as np


def _pure_nash(q_red, q_blue, red, blue, rounds=200):
    """Iterated best response in every state at once, from the given starting actions (indices). Returns the actions
    and, per state, whether they are a pure Nash equilibrium of that state's grid game."""
    rows, n = np.arange(len(red))[:, None], q_red.shape[1]
    for _ in range(rounds):
        new_red = q_red[rows, np.arange(n)[None, :], blue[:, None]].argmax(1)
        new_blue = q_blue[rows, new_red[:, None], np.arange(n)[None, :]].argmax(1)
        if np.array_equal(new_red, red) and np.array_equal(new_blue, blue):
            break
        red, blue = new_red, new_blue
    idx = np.arange(len(red))
    stable = ((q_red[idx, red, blue] >= q_red[rows, np.arange(n)[None, :], blue[:, None]].max(1) - 1e-12)
              & (q_blue[idx, red, blue] >= q_blue[rows, red[:, None], np.arange(n)[None, :]].max(1) - 1e-12))
    return red, blue, stable


def solve_1d(h_red, h_blue, w, w0, discount=0.9, states=401, actions=101, tol=1e-10, max_iter=2000, noise=0.0, nodes=7, low=0.0, high=1.0):
    """Feedback equilibrium of one coordinate. Returns the state grid, each player's action and value per state,
    the number of value-iteration sweeps, and how many states had no pure grid equilibrium. With `noise`, the dog's
    next position gets a Gaussian shock of that standard deviation (clipped to [low, high], [0, 1] like the environment
    by default), and expectations use Gauss-Hermite quadrature with `nodes` points."""
    z, a = np.linspace(low, high, states), np.linspace(0.0, 1.0, actions)
    shock, weight = np.polynomial.hermite_e.hermegauss(nodes) if noise else (np.zeros(1), np.ones(1))
    weight = weight / weight.sum()
    mean = w0 * z[:, None, None] + (1 - w0) * (w * a[None, :, None] + (1 - w) * a[None, None, :])
    nxt = np.clip(mean[..., None] + noise * shock, low, high)
    r_red, r_blue = -((nxt - h_red) ** 2) @ weight, -((nxt - h_blue) ** 2) @ weight
    v_red, v_blue = np.zeros(states), np.zeros(states)
    red, blue = np.full(states, actions - 1), np.zeros(states, dtype=int)
    for sweep in range(1, max_iter + 1):
        q_red = r_red + discount * (np.interp(nxt, z, v_red) @ weight)
        q_blue = r_blue + discount * (np.interp(nxt, z, v_blue) @ weight)
        red, blue, stable = _pure_nash(q_red, q_blue, red, blue)
        idx = np.arange(states)
        new_red, new_blue = q_red[idx, red, blue], q_blue[idx, red, blue]
        change = max(np.abs(new_red - v_red).max(), np.abs(new_blue - v_blue).max())
        v_red, v_blue = new_red, new_blue
        if change < tol:
            break
    return dict(z=z, red=a[red], blue=a[blue], v_red=v_red, v_blue=v_blue, sweeps=sweep, unresolved=int((~stable).sum()))


def solve(houses, w, w0, **options):
    """The 2D feedback equilibrium as one 1D solution per coordinate."""
    (h_red, h_blue) = (np.asarray(h, dtype=float) for h in houses)
    return [solve_1d(h_red[c], h_blue[c], w, w0, **options) for c in range(2)]


def policy(solution, x):
    """Each player's equilibrium action at dog position x (nearest grid state per coordinate)."""
    red, blue = [], []
    for c, part in enumerate(solution):
        j = int(np.abs(part["z"] - x[c]).argmin())
        red.append(part["red"][j])
        blue.append(part["blue"][j])
    return np.array(red), np.array(blue)
