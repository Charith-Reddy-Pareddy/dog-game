"""Nash-Q with continuous moves on the walking game, from the group's week-7 notes.

The deep Nash-Q in doggame.polar_dqn lets each player walk in one of 10 directions. Here a move is any
vector in the player's move set, and one network reads the state and both players' moves and outputs
both players' Q-values, Q(s, a_red, a_blue). Three move encodings ("which coordinate system is
better?"):

- ``cartesian``: the move (vx, vy) inside the disk of radius max_step, given to the network as
  (vx, vy) / max_step and optimized directly, projected back onto the disk;
- ``polar``: the same disk as (r, theta), given to the network as (r / max_step, cos theta,
  sin theta) so that theta wraps around, with r clipped to [0, max_step];
- ``square``: (vx, vy) inside the square of half-side max_step ("could do square as well").

A best response has to be fast. ``best_response`` finds one player's best move against the other's
four ways: a dense grid (the reference), gradient ascent by autograd, gradient ascent by finite
differences, and jumping to the vertex of a quadratic fitted to five evaluations. ``equilibrium``
moves both players at once, each up the gradient of its own Q (simultaneous gradient play), which
is also how the learning target r + discount * Q(s', equilibrium moves) is computed.
"""

import random
import time

import numpy as np
import torch
from torch import nn

from doggame.env import Square, negative_squared_distance_reward
from doggame.nash import solve_stage_nash

ENCODINGS = ("cartesian", "polar", "square")
FEATURES = dict(cartesian=2, polar=3, square=2)


class QNetwork(nn.Module):
    """Both players' Q-values from the state (dog, red, blue) and both players' encoded moves."""

    def __init__(self, encoding="cartesian", hidden=64):
        super().__init__()
        self.encoding = encoding
        self.body = nn.Sequential(nn.Linear(6 + 2 * FEATURES[encoding], hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def features(self, p):
        """Network input for moves in the optimizer's coordinates (normalized by max_step)."""
        if self.encoding == "polar":
            return torch.stack([p[..., 0], torch.cos(p[..., 1]), torch.sin(p[..., 1])], -1)
        return p

    def forward(self, states, p_red, p_blue):
        out = self.body(torch.cat([states, self.features(p_red), self.features(p_blue)], -1))
        return out[..., 0], out[..., 1]


def project(p, encoding):
    """Back onto the move set, in normalized coordinates (unit disk, unit square, or r in [0, 1])."""
    if encoding == "polar":
        return torch.stack([p[..., 0].clamp(0, 1), p[..., 1]], -1)
    if encoding == "square":
        return p.clamp(-1, 1)
    norm = p.norm(dim=-1, keepdim=True).clamp(min=1.0)
    return p / norm


def displacement(p, encoding, max_step):
    """The actual move vector for normalized coordinates p."""
    p = np.asarray(p, dtype=float)
    if encoding == "polar":
        return max_step * p[..., 0:1] * np.stack([np.cos(p[..., 1]), np.sin(p[..., 1])], -1)
    return max_step * p


def random_moves(n, encoding):
    """Uniform moves in the move set, normalized."""
    if encoding == "square":
        return torch.rand(n, 2) * 2 - 1
    r, theta = torch.sqrt(torch.rand(n)), torch.rand(n) * 2 * np.pi
    if encoding == "polar":
        return torch.stack([r, theta], -1)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], -1)


def equilibrium(net, states, iters=30, lr=0.1, start=None, tol=None):
    """Simultaneous gradient play on the stage game Q(s, ., .): each player steps up the gradient of
    its own Q with respect to its own move, then projects. Returns (p_red, p_blue, iterations)."""
    n = states.shape[0]
    p = [torch.zeros(n, 2) if start is None else start[i].clone() for i in range(2)]
    if net.encoding == "polar" and start is None:
        p = [torch.stack([torch.full((n,), 0.5), torch.zeros(n)], -1) for _ in range(2)]
    for it in range(1, iters + 1):
        p = [x.detach().requires_grad_() for x in p]
        q_red, q_blue = net(states, p[0], p[1])
        g_red = torch.autograd.grad(q_red.sum(), p[0], retain_graph=True)[0]
        g_blue = torch.autograd.grad(q_blue.sum(), p[1])[0]
        new = [project(p[0] + lr * g_red, net.encoding).detach(), project(p[1] + lr * g_blue, net.encoding).detach()]
        change = max((new[i] - p[i]).abs().max().item() for i in range(2))
        p = new
        if tol is not None and change < tol:
            return p[0], p[1], it
    return p[0], p[1], iters


def _grid(encoding, n=41):
    if encoding == "polar":
        r, t = np.meshgrid(np.linspace(0, 1, n // 2 + 1), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False))
        return torch.tensor(np.stack([r.ravel(), t.ravel()], -1), dtype=torch.float32)
    axis = np.linspace(-1, 1, n)
    g = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    if encoding == "cartesian":
        g = g[np.linalg.norm(g, axis=1) <= 1 + 1e-9]
    return torch.tensor(g, dtype=torch.float32)


def best_response(net, state, other, player, method="autograd", steps=30, lr=0.1, eps=1e-3, h=0.25):
    """One player's best move against the other's, by ``grid``, ``autograd``, ``finite`` (finite-
    difference gradient ascent) or ``quadratic`` (fit a parabola along each axis through five points,
    jump to its vertex, project; repeat with a smaller spacing). Returns (move, Q, evaluations)."""
    state, other = state.view(1, -1), other.view(1, 2)

    def q(p):  # this player's Q for a batch of its own moves
        s, o = state.expand(len(p), -1), other.expand(len(p), -1)
        return net(s, p, o)[0] if player == 0 else net(s, o, p)[1]

    with torch.no_grad():
        if method == "grid":
            g = _grid(net.encoding)
            values = q(g)
            k = int(values.argmax())
            return g[k], float(values[k]), len(g)
    start = torch.tensor([[0.5, 0.0]]) if net.encoding == "polar" else torch.zeros(1, 2)
    p, evals = start, 0
    if method == "autograd":
        for _ in range(steps):
            p = p.detach().requires_grad_()
            grad = torch.autograd.grad(q(p).sum(), p)[0]
            p, evals = project(p + lr * grad, net.encoding).detach(), evals + 1
    elif method == "finite":
        basis = torch.eye(2) * eps
        for _ in range(steps):
            with torch.no_grad():
                grad = ((q(p + basis) - q(p - basis)) / (2 * eps)).view(1, 2)
            p, evals = project(p + lr * grad, net.encoding), evals + 4
    elif method == "quadratic":
        for _ in range(max(1, steps // 10)):
            with torch.no_grad():
                pts = torch.cat([p, p + torch.tensor([[h, 0.0]]), p - torch.tensor([[h, 0.0]]),
                                 p + torch.tensor([[0.0, h]]), p - torch.tensor([[0.0, h]])])
                f0, fx1, fx2, fy1, fy2 = q(pts).tolist()
            step = []
            for f_plus, f_minus in ((fx1, fx2), (fy1, fy2)):
                curve, slope = (f_plus - 2 * f0 + f_minus) / h ** 2, (f_plus - f_minus) / (2 * h)
                step.append(-slope / curve if curve < 0 else (h if f_plus > f_minus else -h))
            p, evals, h = project(p + torch.tensor([step]), net.encoding), evals + 5, h / 2
    with torch.no_grad():
        return p[0], float(q(p)), evals


def walk(env, d_red, d_blue):
    """One round of the walking game with move vectors (the square move set can exceed max_step)."""
    env.red, env.blue = env.domain.clip(env.red + d_red), env.domain.clip(env.blue + d_blue)
    env.dog = env.domain.clip(env.transition(env.dog, env.red, env.blue, env.w))
    return env.state(), negative_squared_distance_reward(env.dog, env.house_red, env.house_blue)


def train(env, encoding="cartesian", episodes=300, steps=15, batch_size=64, lr=1e-3, target_every=100,
          explore=(1.0, 0.05), seed=0, every=25, on_checkpoint=None):
    """Nash-Q with continuous moves: replay buffer, frozen target network, epsilon-random moves, and
    the target r + discount * Q_target(s', simultaneous-gradient equilibrium of Q_target at s')."""
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    net, target = QNetwork(encoding), QNetwork(encoding)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer, updates = [], 0
    for episode in range(episodes):
        epsilon = explore[0] + episode / max(1, episodes - 1) * (explore[1] - explore[0])
        state = env.reset()
        for _ in range(steps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            if random.random() < epsilon:
                p_red, p_blue = random_moves(1, encoding), random_moves(1, encoding)
            else:
                p_red, p_blue, _ = equilibrium(net, s)
            d = displacement(torch.cat([p_red, p_blue]).numpy(), encoding, env.max_step)
            next_state, rewards = walk(env, d[0], d[1])
            buffer.append((state, p_red[0].numpy(), p_blue[0].numpy(), rewards, next_state))
            state = next_state
            if len(buffer) < batch_size:
                continue
            states, pr, pb, rw, nexts = (torch.as_tensor(np.array(x), dtype=torch.float32)
                                          for x in zip(*random.sample(buffer, batch_size)))
            nr, nb, _ = equilibrium(target, nexts, iters=20)
            with torch.no_grad():
                goal = rw + env.discount * torch.stack(target(nexts, nr, nb), -1)
            loss = nn.functional.mse_loss(torch.stack(net(states, pr, pb), -1), goal)
            optimizer.zero_grad(), loss.backward(), optimizer.step()
            updates += 1
            if updates % target_every == 0:
                target.load_state_dict(net.state_dict())
        if on_checkpoint and (episode + 1) % every == 0:
            on_checkpoint(episode + 1, net)
    return net


def play(env, net, steps=15, start=None):
    """Greedy equilibrium play: both players make the stage-game equilibrium moves of the learned Q.
    ``start`` = (red, blue) positions to begin from instead of the houses. Returns the paths."""
    env.reset()
    if start is not None:
        env.red, env.blue = np.asarray(start[0], dtype=float), np.asarray(start[1], dtype=float)
        env.dog = env.w * env.red + (1 - env.w) * env.blue
    path = [(env.red.copy(), env.blue.copy(), env.dog.copy())]
    for _ in range(steps):
        p_red, p_blue, _ = equilibrium(net, torch.as_tensor(env.state(), dtype=torch.float32).unsqueeze(0))
        d = displacement(torch.cat([p_red, p_blue]).numpy(), net.encoding, env.max_step)
        walk(env, d[0], d[1])
        path.append((env.red.copy(), env.blue.copy(), env.dog.copy()))
    return path


STARTS = [tuple(np.random.default_rng(k).random((2, 2))) for k in range(8)]  # fixed random starts


def corner_distance(env, net, steps=15, start=None):
    """Summed distance of the players' final positions from the one-shot Nash corners."""
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, Square())
    red, blue, _ = play(env, net, steps, start)[-1]
    return float(np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue))


def random_start_distance(env, net, steps=15):
    """The mean corner distance over the eight fixed random starting positions."""
    return float(np.mean([corner_distance(env, net, steps, start) for start in STARTS]))


def benchmark(net, env, states=50, seed=0, **options):
    """Each best-response method against the grid reference at random states and random opponent
    moves: mean shortfall in Q, Q evaluations and milliseconds per best response."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = {m: dict(shortfall=[], relative=[], evals=[], ms=[]) for m in ("grid", "autograd", "finite", "quadratic")}
    for _ in range(states):
        state = torch.as_tensor(np.concatenate([rng.random(2), rng.random(2), rng.random(2)]), dtype=torch.float32)
        other, player = random_moves(1, net.encoding)[0], int(rng.integers(2))
        reference = best_response(net, state, other, player, "grid")[1]
        with torch.no_grad():  # the spread of this player's Q over its moves, to scale the shortfall
            g = _grid(net.encoding)
            s, o = state.view(1, -1).expand(len(g), -1), other.view(1, 2).expand(len(g), -1)
            values = net(s, g, o)[0] if player == 0 else net(s, o, g)[1]
            spread = float(values.max() - values.min())
        for m in out:
            t = time.perf_counter()
            _, value, evals = best_response(net, state, other, player, m, **options)
            out[m]["ms"].append(1000 * (time.perf_counter() - t))
            out[m]["shortfall"].append(reference - value)
            out[m]["relative"].append((reference - value) / spread if spread > 0 else 0.0)
            out[m]["evals"].append(evals)
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in out.items()}


def equilibrium_gap(net, state, p_red, p_blue):
    """How much either player gains by switching to its grid best move against the other's."""
    with torch.no_grad():
        q_red, q_blue = net(state.view(1, -1), p_red.view(1, 2), p_blue.view(1, 2))
    gain_red = best_response(net, state, p_blue, 0, "grid")[1] - float(q_red)
    gain_blue = best_response(net, state, p_red, 1, "grid")[1] - float(q_blue)
    return max(gain_red, gain_blue, 0.0)


def convergence(net, states=30, rates=(0.03, 0.1, 0.3, 1.0, 3.0), iters=500, seed=0):
    """Simultaneous gradient play on the stage game at random states for each step size: how many
    iterations until no move changes by more than 1e-5 (``iters`` if never), and the remaining
    equilibrium gap."""
    rng = np.random.default_rng(seed)
    pts = torch.as_tensor(rng.random((states, 6)), dtype=torch.float32)
    out = {}
    for lr in rates:
        steps, gaps = [], []
        for k in range(states):
            p_red, p_blue, n = equilibrium(net, pts[k:k + 1], iters=iters, lr=lr, tol=1e-5)
            steps.append(n), gaps.append(equilibrium_gap(net, pts[k], p_red[0], p_blue[0]))
        out[lr] = dict(converged=float(np.mean(np.array(steps) < iters)), iterations=float(np.median(steps)),
                       gap=float(np.mean(gaps)))
    return out


TRAJECTORY_STARTS = {"from the houses": None, "houses swapped": "swap", "both in the middle": ((0.5, 0.5), (0.52, 0.48)),
                     "each in the other's corner": ((0.05, 0.95), (0.95, 0.05))}


def trajectories(env, net, steps=15):
    """Greedy equilibrium paths from a few interesting starts (``swap`` puts each player on the
    other's house)."""
    out = {}
    for name, start in TRAJECTORY_STARTS.items():
        if start == "swap":
            start = (env.house_blue, env.house_red)
        out[name] = [[list(map(float, x)) for x in point] for point in play(env, net, steps, start)]
    return out
