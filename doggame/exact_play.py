"""Exact gradient play on feedback policies of one coordinate of the game with inertia.

Each player's objective is its discounted payoff averaged over a grid of starting dog positions, computed exactly by
following the deterministic dog, and each ascends its own gradient (Adam) at the same time. There is no sampling
noise, so this separates the learning dynamics from the noise of policy-gradient estimates. Two policy classes: a
table with one parameter per grid position (linearly interpolated between them), and a small network shared across
positions. Results are scored against Proposition 2's policy and by their largest one-round deviation gain.
"""

import numpy as np
import torch

from doggame.inertia_theory import deviation_gain, explicit_policy, nash_1d

GRID = 41


def _squash(kind):
    if kind == "sigmoid":
        return torch.sigmoid
    return lambda r: r + (r.clamp(0, 1) - r).detach()  # clamp forward, identity gradient (straight-through)


def _policies(kind, squash, hidden, seed):
    torch.manual_seed(seed)
    if kind == "table":
        thetas = [torch.full((GRID,), 0.0 if squash == "sigmoid" else 0.5, requires_grad=True) for _ in range(2)]
        f = _squash(squash)

        def table(theta, x):
            pos = (x * (GRID - 1)).clamp(0, GRID - 1)
            i = pos.floor().long().clamp(max=GRID - 2)
            a = f(theta)
            return a[i] * (1 - (pos - i)) + a[i + 1] * (pos - i)
        return [[t] for t in thetas], [lambda x, t=t: table(t, x) for t in thetas]
    nets = [torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)) for _ in range(2)]
    f = _squash(squash)
    return [list(n.parameters()) for n in nets], [lambda x, n=n: f(n(x[:, None])[:, 0]) for n in nets]


def play(h_red, h_blue, w, w0, kind="table", squash="sigmoid", hidden=16, seed=0, steps=3000, discount=0.9, rounds=60, lr=None):
    """Run exact gradient play; returns the learned actions on the grid of dog positions (red, blue)."""
    params, policy = _policies(kind, squash, hidden, seed)
    opts = [torch.optim.Adam(p, lr=lr or (0.05 if kind == "table" else 0.01)) for p in params]
    grid, houses = torch.linspace(0, 1, GRID), (h_red, h_blue)
    for _ in range(steps):
        x, values = grid, [0.0, 0.0]
        for k in range(rounds):
            x = w0 * x + (1 - w0) * (w * policy[0](x) + (1 - w) * policy[1](x))
            values = [v - discount ** k * (x - h) ** 2 for v, h in zip(values, houses)]
        grads = [torch.autograd.grad(-values[i].mean(), params[i], retain_graph=i == 0) for i in range(2)]  # each its own payoff
        for i in range(2):  # then both step at once
            for p, g in zip(params[i], grads[i]):
                p.grad = g
            opts[i].step()
    with torch.no_grad():
        return tuple(np.clip(p(grid).numpy().astype(float), 0, 1) for p in policy)


def score(red, blue, h_red, h_blue, w, w0, discount=0.9):
    """Largest action gap to Proposition 2's policy and to holding, and the largest one-round deviation gain."""
    grid = np.linspace(0, 1, GRID)
    a_red, a_blue, _ = nash_1d(h_red, h_blue, w)
    e_red, e_blue = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0, discount)(grid)
    act = lambda x: (np.interp(x, grid, red), np.interp(x, grid, blue))
    gain = max(max(deviation_gain(act, h_red, w, w0, x, True, discount, grid=1001), deviation_gain(act, h_blue, w, w0, x, False, discount, grid=1001))
               for x in grid)
    return dict(gap_explicit=float(max(np.abs(red - e_red).max(), np.abs(blue - e_blue).max())),
                gap_holding=float(max(np.abs(red - a_red).max(), np.abs(blue - a_blue).max())), gain=float(gain))


def play_many(houses, weights, w0, kind="table", hidden=16, seed=0, steps=3000, discount=0.9, rounds=60):
    """Exact gradient play with N players on one coordinate (sigmoid policies): a table of actions per player, or one
    small network per player. Returns the learned actions on the grid, shape (N, GRID)."""
    torch.manual_seed(seed)
    n, grid = len(houses), torch.linspace(0, 1, GRID)
    if kind == "table":
        params = [[torch.zeros(GRID, requires_grad=True)] for _ in range(n)]

        def policy(i, x):
            pos = (x * (GRID - 1)).clamp(0, GRID - 1)
            j = pos.floor().long().clamp(max=GRID - 2)
            a = torch.sigmoid(params[i][0])
            return a[j] * (1 - (pos - j)) + a[j + 1] * (pos - j)
    else:
        nets = [torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)) for _ in range(n)]
        params = [list(net.parameters()) for net in nets]
        policy = lambda i, x: torch.sigmoid(nets[i](x[:, None])[:, 0])
    opts = [torch.optim.Adam(p, lr=0.05 if kind == "table" else 0.01) for p in params]
    weights_t = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    for _ in range(steps):
        x, values = grid, [0.0] * n
        for k in range(rounds):
            x = w0 * x + (1 - w0) * sum(weights_t[i] * policy(i, x) for i in range(n))
            values = [v - discount ** k * (x - h) ** 2 for v, h in zip(values, houses)]
        grads = [torch.autograd.grad(-values[i].mean(), params[i], retain_graph=i < n - 1) for i in range(n)]
        for i in range(n):
            for p, g in zip(params[i], grads[i]):
                p.grad = g
            opts[i].step()
    with torch.no_grad():
        return np.stack([policy(i, grid).numpy().astype(float) for i in range(n)])


def deviation_gain_many(actions, houses, weights, w0, discount=0.9, states=21, grid=201, rounds=200):
    """The most any player gains at any of `states` dog positions by changing its action for one round and then
    following `actions` (an (N, GRID) table, interpolated) again."""
    xs, houses, weights = np.linspace(0, 1, GRID), np.asarray(houses, dtype=float), np.asarray(weights, dtype=float)
    act = lambda z: np.stack([np.interp(z, xs, a) for a in actions])
    best = 0.0
    for x in np.linspace(0, 1, states):
        now = act(np.array([x]))[:, 0]
        for i in range(len(houses)):
            own = np.append(np.linspace(0, 1, grid), now[i])
            z = w0 * x + (1 - w0) * (now @ weights - weights[i] * now[i] + weights[i] * own)
            total = -(z - houses[i]) ** 2
            for k in range(1, rounds):
                z = w0 * z + (1 - w0) * weights @ act(z)
                total -= discount ** k * (z - houses[i]) ** 2
            best = max(best, float(total[:-1].max() - total[-1]))
    return best
