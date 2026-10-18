"""Learning in the sticky-price duopoly (doggame.sticky_duopoly): does a learner find the feedback
equilibrium, or does it settle on the static Cournot output?

``play`` is exact gradient play: each firm has a deterministic policy from the price to its output
(a table on a price grid, linearly interpolated, or a small network, both squashed into [0, Q_MAX]),
its objective is its discounted profit averaged over evenly spread starting prices, computed exactly
by following the price, and both firms step up their own gradients at once (Adam). With ``fix``,
best-response regression instead: every 25 steps it computes every firm's exact best one-round
output at each grid price (continuation by the current policies) and fits the policy to those
outputs, so every price counts equally instead of in proportion to how strongly its gradient pulls.
"""

import numpy as np
import torch

from doggame.sticky_duopoly import A, C, Q_MAX, linear_equilibrium

GRID = 41


def _policies(kind, hidden, seed):
    torch.manual_seed(seed)
    if kind == "table":
        thetas = [torch.zeros(GRID, requires_grad=True) for _ in range(2)]

        def table(theta, p):
            pos = (p / A * (GRID - 1)).clamp(0, GRID - 1)
            i = pos.floor().long().clamp(max=GRID - 2)
            q = Q_MAX * torch.sigmoid(theta)
            return q[i] * (1 - (pos - i)) + q[i + 1] * (pos - i)
        return [[t] for t in thetas], [lambda p, t=t: table(t, p) for t in thetas]
    nets = [torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)) for _ in range(2)]
    return [list(n.parameters()) for n in nets], [lambda p, n=n: Q_MAX * torch.sigmoid(n(p[:, None])[:, 0]) for n in nets]


def _profits(policy, w0, discount, rounds, prices):
    p, total = prices, [0.0, 0.0]
    for k in range(rounds):
        q = [policy[0](p), policy[1](p)]
        p = (w0 * p + (1 - w0) * (A - q[0] - q[1])).clamp(0, A)
        total = [t + discount ** k * (p * qi - C * qi - qi ** 2 / 2) for t, qi in zip(total, q)]
    return total


def play(w0, kind="table", hidden=16, seed=0, steps=3000, discount=0.9, rounds=60, lr=None, fix=False):
    """Exact gradient play, or with ``fix`` best-response regression. Returns both firms' outputs on the
    price grid, shape (2, GRID)."""
    params, policy = _policies(kind, hidden, seed)
    opts = [torch.optim.Adam(p, lr=lr or (0.05 if kind == "table" else 0.01)) for p in params]
    prices = torch.linspace(0, A, GRID)
    for t in range(steps):
        if fix and t % 25 == 0:  # refresh the per-price best responses the policies are fitted to
            with torch.no_grad():
                goal = torch.as_tensor(best_outputs([lambda x, f=f: f(torch.as_tensor(x, dtype=torch.float32)).numpy() for f in policy],
                                                    w0, discount), dtype=torch.float32)
        if fix:
            losses = [((policy[i](prices) - goal[i]) ** 2).mean() for i in range(2)]
            grads = [torch.autograd.grad(losses[i], params[i]) for i in range(2)]
        else:
            profits = _profits(policy, w0, discount, rounds, prices)
            grads = [torch.autograd.grad(-profits[i].mean(), params[i], retain_graph=i == 0) for i in range(2)]
        for i in range(2):
            for p, g in zip(params[i], grads[i]):
                p.grad = g
            opts[i].step()
    with torch.no_grad():
        return np.stack([policy[i](prices).numpy().astype(float) for i in range(2)])


def _continue(policy, x, w0, discount, rounds, firm):
    total = np.zeros_like(x)
    for k in range(1, rounds):
        q = [policy[0](x), policy[1](x)]
        nx = np.clip(w0 * x + (1 - w0) * (A - q[0] - q[1]), 0, A)
        total += discount ** k * (nx * q[firm] - C * q[firm] - q[firm] ** 2 / 2)
        x = nx
    return total


def best_outputs(policy, w0, discount=0.9, grid=201, rounds=120):
    """Each firm's best one-round output at every grid price, the rival and the continuation following
    ``policy`` (a pair of numpy functions price -> output)."""
    prices, outs = np.linspace(0, A, GRID), np.linspace(0, Q_MAX, grid)
    best = np.zeros((2, GRID))
    for k, p in enumerate(prices):
        for firm in range(2):
            rival = float(policy[1 - firm](np.array([p]))[0])
            x = np.clip(w0 * p + (1 - w0) * (A - outs - rival), 0, A)
            value = x * outs - C * outs - outs ** 2 / 2 + _continue(policy, x, w0, discount, rounds, firm)
            best[firm, k] = outs[value.argmax()]
    return best


def score(outputs, w0, discount=0.9, grid=401, rounds=200):
    """The learned policies (outputs on the price grid) against the linear feedback equilibrium and the
    static Cournot output: largest output gaps over prices, and the largest one-round gain of either firm."""
    prices = np.linspace(0, A, GRID)
    policy = [lambda x, o=o: np.interp(x, prices, o) for o in outputs]
    a, b, _ = linear_equilibrium(w0, discount)
    feedback = np.clip(a + b * prices, 0, Q_MAX)
    cournot = (A - C) / 4
    outs, gain = np.linspace(0, Q_MAX, grid), 0.0
    for p in prices[2:-2]:  # inner prices, where the equilibrium is interior
        for firm in range(2):
            rival, now = float(policy[1 - firm](np.array([p]))[0]), float(policy[firm](np.array([p]))[0])
            trial = np.append(outs, now)
            x = np.clip(w0 * p + (1 - w0) * (A - trial - rival), 0, A)
            value = x * trial - C * trial - trial ** 2 / 2 + _continue(policy, x, w0, discount, rounds, firm)
            gain = max(gain, float(value[:-1].max() - value[-1]))
    return dict(to_feedback=float(np.abs(outputs - feedback).max()), to_cournot=float(np.abs(outputs - cournot).max()), gain=gain)
