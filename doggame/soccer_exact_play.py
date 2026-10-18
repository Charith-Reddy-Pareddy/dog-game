"""Exact (noise-free) gradient play on the soccer game (doggame.soccer): a second game for the finding that a table of
actions finds the feedback equilibrium and a shared network does not (doggame.exact_play does it for the dog game).

Both players' values are computed exactly from the transition tables (60 unrolled Bellman steps) and each follows the
exact gradient of its own objective, the value averaged over every state (as if each game could start anywhere), with
Adam. Progress is scored by the exact duality gap at each state, which is 0 only at a Markov-perfect equilibrium.
With ``fix``, equilibrium regression instead: every ``refresh`` steps each state's stage game is solved exactly and
both policies are fitted to those strategies, every state counting equally. The continuation values are those of the
current policies (``fix="policy"``, a policy-iteration step, which need not converge in zero-sum games: Van der Wal
1978), or a value iterate that takes one Shapley step per refresh (``fix="value"``, approximate value iteration as in
Perolat et al. 2015), which converges whatever the policies do.
"""

import numpy as np
import torch

from doggame.exact_nash import zero_sum_lp_nash
from doggame.soccer import N_ACTIONS


def duality_gap(p0, p1, game, discount=0.9, iters=400):
    """Per state: player 0's best-response value against p1 minus player 1's best-response value against p0."""
    v0, v1 = np.zeros(game.n_states + 1), np.zeros(game.n_states + 1)
    for _ in range(iters):
        v0[:-1] = np.einsum("sab,sb->sa", game.reward + discount * v0[game.next_state], p1).max(1)
        v1[:-1] = np.einsum("sa,sab->sb", p0, game.reward + discount * v1[game.next_state]).min(1)
    return v0[:-1] - v1[:-1]


def stage_equilibrium(q):
    """Per state, an equilibrium of the stage game q (states x actions x actions, player 0 maximizes): a pure
    saddle point where there is one, else the minimax linear program."""
    lower, upper = q.min(2), q.max(1)
    p0, p1 = np.eye(N_ACTIONS)[lower.argmax(1)], np.eye(N_ACTIONS)[upper.argmin(1)]
    for s in np.flatnonzero(lower.max(1) < upper.min(1) - 1e-9):
        p0[s], p1[s], _ = zero_sum_lp_nash(q[s])
    return p0, p1


def _stage_values(p0, p1, game, discount=0.9, iters=200):
    v = np.zeros(game.n_states + 1)
    for _ in range(iters):
        v[:-1] = np.einsum("sa,sab,sb->s", p0, game.reward + discount * v[game.next_state], p1)
    return game.reward + discount * v[game.next_state]


def play(game, policy="table", steps=2000, lr=None, seed=0, discount=0.9, rounds=60, every=100, hidden=64, fix=None, refresh=25):
    """Exact gradient play with a table of logits per state or one network per player. Returns rows
    (step, mean gap, largest gap, largest gap at the starting states)."""
    torch.manual_seed(seed)
    n, starts = game.n_states, game.start_states()
    reward, nxt = torch.as_tensor(game.reward, dtype=torch.float32), torch.as_tensor(game.next_state)
    if policy == "table":
        logits = [torch.zeros(n, N_ACTIONS, requires_grad=True) for _ in range(2)]
        params, probs = [[t] for t in logits], lambda i: torch.softmax(logits[i], 1)
    else:
        feats = torch.as_tensor(game.features(range(n)))
        nets = [torch.nn.Sequential(torch.nn.Linear(5, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                    torch.nn.Linear(hidden, N_ACTIONS)) for _ in range(2)]
        params, probs = [list(m.parameters()) for m in nets], lambda i: torch.softmax(nets[i](feats), 1)
    opts = [torch.optim.Adam(p, lr=lr or (0.05 if policy == "table" else 0.003)) for p in params]
    rows, v = [], np.zeros(n + 1)
    for t in range(steps + 1):
        if t % every == 0:
            gap = duality_gap(probs(0).detach().numpy().astype(float), probs(1).detach().numpy().astype(float), game, discount)
            rows.append((t, float(gap.mean()), float(gap.max()), float(gap[starts].max())))
        if t == steps:
            break
        if fix:
            if t % refresh == 0:
                if fix == "policy":
                    q = _stage_values(*[probs(i).detach().numpy().astype(float) for i in range(2)], game, discount)
                else:
                    q = game.reward + discount * v[game.next_state]
                goal = stage_equilibrium(q)
                v[:-1] = np.einsum("sa,sab,sb->s", goal[0], q, goal[1])
                goal = [torch.as_tensor(g, dtype=torch.float32) for g in goal]
            grads = [torch.autograd.grad(-(goal[i] * torch.log(probs(i) + 1e-12)).sum(1).mean(), params[i]) for i in range(2)]  # cross-entropy
        else:
            v, p0, p1 = torch.zeros(n + 1), probs(0), probs(1)
            for _ in range(rounds):
                v = torch.cat([torch.einsum("sa,sab,sb->s", p0, reward + discount * v[nxt], p1), torch.zeros(1)])
            value = v[:-1].mean()
            grads = torch.autograd.grad(-value, params[0], retain_graph=True), torch.autograd.grad(value, params[1])
        for opt, ps, gr in zip(opts, params, grads):
            for p, g in zip(ps, gr):
                p.grad = g
            opt.step()
    return rows


def fit(game, solution, hidden=64, steps=4000, lr=0.003, seed=0):
    """Fit one network per player (as in ``play``) to the solved equilibrium by cross-entropy: can the network
    represent the equilibrium at all? Returns the share of states where its most likely action is the
    equilibrium's, and the mean and largest duality gap."""
    torch.manual_seed(seed)
    feats = torch.as_tensor(game.features(range(game.n_states)))
    nets = [torch.nn.Sequential(torch.nn.Linear(5, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, N_ACTIONS)) for _ in range(2)]
    goal = [torch.as_tensor(p, dtype=torch.float32) for p in (solution.policy0, solution.policy1)]
    opt = torch.optim.Adam([p for m in nets for p in m.parameters()], lr=lr)
    for _ in range(steps):
        loss = sum(-(g * torch.log_softmax(m(feats), 1)).sum(1).mean() for m, g in zip(nets, goal))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = [torch.softmax(m(feats), 1).numpy().astype(float) for m in nets]
    gap = duality_gap(*probs, game)
    agree = np.mean([(p.argmax(1) == g.numpy().argmax(1)).mean() for p, g in zip(probs, goal)])
    return dict(agree=float(agree), mean_gap=float(gap.mean()), max_gap=float(gap.max()))
