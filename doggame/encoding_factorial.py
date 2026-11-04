"""Why do (vx, vy) and (r, theta) moves give different walking-game paths? A 2 x 2 x 2 study of continuous-move Nash-Q.

Three things vary, one at a time; the game, network size, learning rule and episodes do not:

- ``inp``: what the network sees, the move (vx, vy) or (r, cos theta, sin theta);
- ``opt``: the coordinates the stage-game search steps in, (vx, vy) or (r, theta);
- ``region``: the moves allowed, a disk or a square of half-side max_step.

Each trained network is played twice, with the gradient search used in training and with an exhaustive grid search.
With ``warm`` the polar search first tries 8 directions at training and play time, instead of always starting at
theta = 0. See docs/ENCODING_NOTES.md for the results, which ``summary`` recomputes from the saved runs.

Run one cell with `python3 -m doggame.encoding_factorial inp opt region seed`; it appends to OUTPUT. Run the test of
the polar search's starting angle with `python3 -m doggame.encoding_factorial base|warm seed`; it appends to OUTPUT_WARM.
"""

import itertools
import json
import random
import sys

import numpy as np
import torch
from torch import nn

from doggame.dog_game_study import WALK_HOUSES
from doggame.env import Square, negative_squared_distance_reward
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv

OUTPUT = "results/encoding_factorial.jsonl"
OUTPUT_WARM = "results/encoding_warm_start.jsonl"
STARTS = [None, "swap", ((0.5, 0.5), (0.52, 0.48)), ((0.05, 0.95), (0.95, 0.05))]
STARTS += [tuple(np.random.default_rng(k).random((2, 2))) for k in range(8)]
PI = np.pi
START_ANGLES = {"0,0": (0, 0), "pi,pi": (PI, PI), "pi/2,pi/2": (PI / 2, PI / 2), "-pi/2,-pi/2": (-PI / 2, -PI / 2), "0,pi": (0, PI), "pi,0": (PI, 0)}


class Net(nn.Module):
    """Both players' Q-values from the state and both players' moves, given in search coordinates p."""

    def __init__(self, inp, opt, region, hidden=64):
        super().__init__()
        self.inp, self.opt, self.region = inp, opt, region
        width = 2 if inp == "cart" else 3
        self.body = nn.Sequential(nn.Linear(6 + 2 * width, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))

    def reach(self, theta):
        """How far the region extends in direction theta."""
        return torch.ones_like(theta) if self.region == "disk" else 1 / torch.cos(theta).abs().maximum(torch.sin(theta).abs())

    def vec(self, p):
        """The move vector in units of max_step."""
        if self.opt == "cart":
            return p
        r = p[..., 0] * self.reach(p[..., 1])
        return torch.stack([r * torch.cos(p[..., 1]), r * torch.sin(p[..., 1])], -1)

    def from_vec(self, v):
        """Search coordinates of move vectors."""
        if self.opt == "cart":
            return v
        theta = torch.atan2(v[..., 1], v[..., 0])
        return torch.stack([v.norm(dim=-1) / self.reach(theta), theta], -1)

    def features(self, p):
        if self.inp == "cart":
            return self.vec(p)
        if self.opt == "polar":
            return torch.stack([p[..., 0] * self.reach(p[..., 1]), torch.cos(p[..., 1]), torch.sin(p[..., 1])], -1)
        r = (p.pow(2).sum(-1) + 1e-10).sqrt()
        return torch.stack([r, p[..., 0] / r, p[..., 1] / r], -1)

    def project(self, p):
        if self.opt == "polar":
            return torch.stack([p[..., 0].clamp(0, 1), p[..., 1]], -1)
        return p.clamp(-1, 1) if self.region == "square" else p / p.norm(dim=-1, keepdim=True).clamp(min=1.0)

    def forward(self, states, a, b):
        out = self.body(torch.cat([states, self.features(a), self.features(b)], -1))
        return out[..., 0], out[..., 1]

    def sample(self, n):
        """Uniform moves in the region."""
        if self.region == "square":
            return self.from_vec(torch.rand(n, 2) * 2 - 1)
        r, theta = torch.sqrt(torch.rand(n)), torch.rand(n) * 2 * np.pi
        return self.from_vec(torch.stack([r * torch.cos(theta), r * torch.sin(theta)], -1))

    def grid(self, k=33):
        axis = np.linspace(-1, 1, k)
        g = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        if self.region == "disk":
            g = g[np.linalg.norm(g, axis=1) <= 1 + 1e-9]
        return self.from_vec(torch.tensor(g, dtype=torch.float32))


def angles_of(net, states, warm):
    """Each player's starting angle for the polar search: 0, or with ``warm`` the best of 8 against the other standing still."""
    n = len(states)
    if not warm:
        return torch.zeros(n), torch.zeros(n)
    angles = torch.arange(8) * 2 * np.pi / 8
    with torch.no_grad():
        s, still = states.repeat_interleave(8, 0), torch.zeros(8 * n, 2)
        moves = torch.stack([torch.full((8 * n,), 0.5), angles.repeat(n)], -1)
        return angles[net(s, moves, still)[0].view(n, 8).argmax(1)], angles[net(s, still, moves)[1].view(n, 8).argmax(1)]


def gradient_play(net, states, iters=30, lr=0.1, warm=False, angles=None):
    """Simultaneous gradient play on the stage game Q(s, ., .), projected onto the region. ``angles`` fixes the polar start."""
    n = len(states)
    if net.opt == "cart":
        p = [torch.zeros(n, 2), torch.zeros(n, 2)]
    else:
        start = angles_of(net, states, warm) if angles is None else [torch.full((n,), float(t)) for t in angles]
        p = [torch.stack([torch.full((n,), 0.5), t], -1) for t in start]
    for _ in range(iters):
        p = [x.detach().requires_grad_() for x in p]
        q_a, q_b = net(states, p[0], p[1])
        g_a = torch.autograd.grad(q_a.sum(), p[0], retain_graph=True)[0]
        g_b = torch.autograd.grad(q_b.sum(), p[1])[0]
        p = [net.project(p[0] + lr * g_a).detach(), net.project(p[1] + lr * g_b).detach()]
    return p


def grid_play(net, state, rounds=6):
    """Alternating exact best responses over a dense grid of moves, from standing still."""
    g, a = net.grid(), net.from_vec(torch.zeros(1, 2))
    b, s = a.clone(), state.expand(len(g), -1)
    with torch.no_grad():
        for _ in range(rounds):
            a = g[net(s, g, b.expand(len(g), -1))[0].argmax()].unsqueeze(0)
            b = g[net(s, a.expand(len(g), -1), g)[1].argmax()].unsqueeze(0)
    return [a, b]


def walk(env, d_a, d_b):
    env.red, env.blue = env.domain.clip(env.red + d_a), env.domain.clip(env.blue + d_b)
    env.dog = env.domain.clip(env.transition(env.dog, env.red, env.blue, env.w))
    return env.state(), negative_squared_distance_reward(env.dog, env.house_red, env.house_blue)


def train(env, inp, opt, region, seed, warm=False, episodes=300, steps=15, batch=64, lr=1e-3, target_every=100):
    """Nash-Q as in doggame.continuous_q (archived): replay buffer, frozen target, epsilon falling from 1 to 0.05."""
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    net, target = Net(inp, opt, region), Net(inp, opt, region)
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer, updates = [], 0
    for episode in range(episodes):
        epsilon = 1.0 + episode / (episodes - 1) * (0.05 - 1.0)
        state = env.reset()
        for _ in range(steps):
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            a, b = (net.sample(1), net.sample(1)) if random.random() < epsilon else gradient_play(net, s, warm=warm)
            d = net.vec(torch.cat([a, b])).detach().numpy() * env.max_step
            following, rewards = walk(env, d[0], d[1])
            buffer.append((state, a[0].numpy(), b[0].numpy(), rewards, following))
            state = following
            if len(buffer) < batch:
                continue
            s_b, a_b, b_b, r_b, n_b = (torch.as_tensor(np.array(x), dtype=torch.float32)
                                       for x in zip(*random.sample(buffer, batch)))
            next_a, next_b = gradient_play(target, n_b, iters=20, warm=warm)
            with torch.no_grad():
                goal = r_b + env.discount * torch.stack(target(n_b, next_a, next_b), -1)
            loss = nn.functional.mse_loss(torch.stack(net(s_b, a_b, b_b), -1), goal)
            optimizer.zero_grad(), loss.backward(), optimizer.step()
            updates += 1
            if updates % target_every == 0:
                target.load_state_dict(net.state_dict())
    return net


def play(env, net, start, choose, steps=15):
    """Both players' positions over ``steps`` rounds of greedy play. Returns round x player x 2."""
    env.reset()
    if start is not None:
        red, blue = (env.house_blue, env.house_red) if isinstance(start, str) else start
        env.red, env.blue = np.array(red, dtype=float), np.array(blue, dtype=float)
        env.dog = env.w * env.red + (1 - env.w) * env.blue
    path = [[env.red.tolist(), env.blue.tolist()]]
    for _ in range(steps):
        a, b = choose(net, torch.as_tensor(env.state(), dtype=torch.float32).unsqueeze(0))
        d = net.vec(torch.cat([a, b])).detach().numpy() * env.max_step
        walk(env, d[0], d[1])
        path.append([env.red.tolist(), env.blue.tolist()])
    return path


def walking_env():
    return PolarDogGameEnv(*WALK_HOUSES, w=0.5, max_step=0.2, discount=0.9)


def corners(env):
    return solve_stage_nash(env.house_red, env.house_blue, env.w, Square())[:2]


def corner_distances(paths, nash):
    """Summed distance of the final positions from the Nash corners, per start."""
    return [float(sum(np.linalg.norm(np.array(p[-1][i]) - nash[i]) for i in range(2))) for p in paths]


def exact_path(start, corner, region, step=0.2, rounds=15):
    """No learning: walk to the corner at full speed. In a disk that is a straight line, in a square each coordinate on its own."""
    p, corner, out = np.array(start, dtype=float), np.array(corner, dtype=float), []
    out.append(p.copy())
    for _ in range(rounds):
        v = corner - p
        p = p + (np.clip(v, -step, step) if region == "square" else v * min(1, step / (np.linalg.norm(v) + 1e-12)))
        out.append(p.copy())
    return np.array(out)


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def cell_paths(rows, key):
    """Mean path of each cell, over seeds: {(inp, opt, region): start x round x player x 2}."""
    cells = {(r["inp"], r["opt"], r["region"]) for r in rows}
    return {c: np.mean([r[key] for r in rows if (r["inp"], r["opt"], r["region"]) == c], 0) for c in cells}


def shares(paths):
    """Each factor's and interaction's share of the variation between the eight cell mean paths (balanced 2 x 2 x 2 ANOVA)."""
    grand = np.mean(list(paths.values()), 0)
    total = sum(((m - grand) ** 2).sum() for m in paths.values())
    names, first = ("input", "search", "region"), ("cart", "cart", "disk")
    out = {}
    for size in (1, 2, 3):
        for chosen in itertools.combinations(range(3), size):
            contrast = sum(np.prod([1 if c[i] == first[i] else -1 for i in chosen]) * m for c, m in paths.items())
            out[" x ".join(names[i] for i in chosen)] = float((contrast ** 2).sum() / 8 / total)
    return out


def gap(a, b):
    """Mean distance between two sets of paths, over starts, rounds and players."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def summary(rows, warm_rows):
    """The numbers quoted in docs/ENCODING_NOTES.md."""
    env = walking_env()
    nash = corners(env)
    played, by_grid = cell_paths(rows, "gradient"), cell_paths(rows, "grid")
    cart, polar, square = ("cart", "cart", "disk"), ("polar", "polar", "disk"), ("cart", "cart", "square")
    polar_square = ("polar", "polar", "square")
    begin = [(env.house_red, env.house_blue), (env.house_blue, env.house_red)] + [s for s in STARTS[2:]]
    exact = {r: np.array([[exact_path(s[i], nash[i], r) for i in range(2)] for s in begin]).transpose(0, 2, 1, 3)
             for r in ("disk", "square")}
    fixed = [r for r in warm_rows if r["mode"] == "warm"]
    return dict(shares=shares(played), polar_gap=gap(played[cart], played[polar]), polar_gap_square=gap(played[square], played[polar_square]),
                region_gap=gap(played[cart], played[square]), polar_region_gap=gap(played[polar], played[polar_square]),
                exact_region_gap=gap(exact["disk"], exact["square"]), grid_polar_gap=gap(by_grid[cart], by_grid[polar]),
                warm_gap=gap(np.mean([r["paths"] for r in fixed], 0), played[cart]),
                warm_seeds_under_01=sum(r["warm"][0] < 0.1 for r in fixed), warm_seeds=len(fixed))


def run_cell(inp, opt, region, seed):
    env = walking_env()
    net = train(env, inp, opt, region, seed)
    row = dict(inp=inp, opt=opt, region=region, seed=seed)
    for name, choose in (("gradient", gradient_play), ("grid", grid_play)):
        row[name] = [play(env, net, s, choose) for s in STARTS]
        row[name + "_corner"] = corner_distances(row[name], corners(env))
    return row


def run_start_angles(mode, seed):
    """A polar network trained as before (``base``) or with the warm start, played from fixed starting angles and warm."""
    env = walking_env()
    nash = corners(env)
    net = train(env, "polar", "polar", "disk", seed, warm=mode == "warm")

    def score(paths):  # the mean corner distance, then red's and blue's final distances from the houses
        end = paths[0][-1]
        return [float(np.mean(corner_distances(paths, nash)))] + [float(np.linalg.norm(np.array(end[i]) - nash[i])) for i in range(2)]

    row = dict(mode=mode, seed=seed)
    for name, angles in START_ANGLES.items():
        row["start " + name] = score([play(env, net, s, lambda n, x, angles=angles: gradient_play(n, x, angles=angles)) for s in STARTS])
    row["paths"] = [play(env, net, s, lambda n, x: gradient_play(n, x, warm=True)) for s in STARTS]
    row["warm"] = score(row["paths"])
    return row


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = sys.argv[1:]
    row = run_cell(args[0], args[1], args[2], int(args[3])) if len(args) == 4 else run_start_angles(args[0], int(args[1]))
    with open(OUTPUT if len(args) == 4 else OUTPUT_WARM, "a") as f:
        f.write(json.dumps(row) + "\n")
