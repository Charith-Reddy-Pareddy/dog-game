"""Exact (noise-free) gradient play on the soccer game (doggame.soccer): a second game for the finding that a table of
actions finds the feedback equilibrium and a shared network does not (doggame.exact_play does it for the dog game).

Both players' values are computed exactly from the transition tables (60 unrolled Bellman steps) and each follows the
exact gradient of its own objective, the value averaged over every state (as if each game could start anywhere), with
Adam. Progress is scored by the exact duality gap at each state, which is 0 only at a Markov-perfect equilibrium.
"""

import numpy as np
import torch

from doggame.soccer import N_ACTIONS


def duality_gap(p0, p1, game, discount=0.9, iters=400):
    """Per state: player 0's best-response value against p1 minus player 1's best-response value against p0."""
    v0, v1 = np.zeros(game.n_states + 1), np.zeros(game.n_states + 1)
    for _ in range(iters):
        v0[:-1] = np.einsum("sab,sb->sa", game.reward + discount * v0[game.next_state], p1).max(1)
        v1[:-1] = np.einsum("sa,sab->sb", p0, game.reward + discount * v1[game.next_state]).min(1)
    return v0[:-1] - v1[:-1]


def play(game, policy="table", steps=2000, lr=None, seed=0, discount=0.9, rounds=60, every=100, hidden=64):
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
    rows = []
    for t in range(steps + 1):
        if t % every == 0:
            gap = duality_gap(probs(0).detach().numpy().astype(float), probs(1).detach().numpy().astype(float), game, discount)
            rows.append((t, float(gap.mean()), float(gap.max()), float(gap[starts].max())))
        if t == steps:
            break
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
