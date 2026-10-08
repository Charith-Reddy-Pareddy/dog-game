"""The dog-game runs behind dog_game_results.pdf, over several seeds.

- convergence: policy-gradient self-play, 5 house setups x 3 network
  architectures, scored by exploitability against the analytical equilibrium.
- walking: the angle-radius game, where players walk instead of jump. A
  REINFORCE policy and the 10-direction Nash-Q network are scored by how far
  the players end up from the one-shot Nash corners.

Each finished run is appended to a jsonl file, and a rerun skips what is
already there. Run with `python3 -m doggame.dog_game_study [seeds]`.
"""

import json
import os
import random
import sys

import numpy as np
import torch

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS, run_convergence_study
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.polar_agent import PolarPolicy, final_positions, train_polar
from doggame.polar_dqn import NashQNetwork, play_greedy, train_nash_dqn

ARCHITECTURES = ("separate", "shared", "partial")
WALK_GAMES = 10  # sampled games per score
WALK_HOUSES = ((0.88, 0.12), (0.3, 0.65))


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["kind"], r["architecture"], r["seed"], r["config"]) for r in map(json.loads, f)}


def _save(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_convergence(seeds, path):
    done = _done(path)
    for seed in seeds:
        for architecture in ARCHITECTURES:
            for index, config in enumerate(DEFAULT_CONFIGS):
                if ("convergence", architecture, seed, index) in done:
                    continue
                (result,) = run_convergence_study([config], architecture, seed=seed)
                gap = max(result["red_exploitability"], result["blue_exploitability"])
                _save(path, dict(kind="convergence", architecture=architecture, seed=seed, config=index, exploitability=gap))


def _walk_distance(env, play):
    """Mean over sampled games of the red plus blue distance to the one-shot Nash corners."""
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, Square())
    total = 0.0
    for _ in range(WALK_GAMES):
        red, blue = play()
        total += np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)
    return total / WALK_GAMES


def run_walking(seeds, path):
    done = _done(path)
    env = PolarDogGameEnv(*WALK_HOUSES, w=0.5, max_step=0.2)
    for seed in seeds:
        if ("walking", "reinforce", seed, 0) not in done:
            torch.manual_seed(seed)
            policy = PolarPolicy(seed=seed)
            before = _walk_distance(env, lambda: final_positions(env, policy))
            train_polar(env, policy, episodes=300, seed=seed)
            after = _walk_distance(env, lambda: final_positions(env, policy))
            _save(path, dict(kind="walking", architecture="reinforce", seed=seed, config=0, before=before, after=after))
        if ("walking", "nash-q", seed, 0) not in done:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            untrained = NashQNetwork()
            before = _walk_distance(env, lambda: play_greedy(env, untrained))
            net = train_nash_dqn(env, episodes=60, seed=seed)
            after = _walk_distance(env, lambda: play_greedy(env, net))
            _save(path, dict(kind="walking", architecture="nash-q", seed=seed, config=0, before=before, after=after))


if __name__ == "__main__":
    seeds = range(int(sys.argv[1])) if len(sys.argv) > 1 else range(5)
    os.makedirs("results", exist_ok=True)
    run_convergence(seeds, "results/dog_game_runs.jsonl")
    run_walking(seeds, "results/dog_game_runs.jsonl")
