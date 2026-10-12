"""Paths of the two players and the dog in the angle-radius game, before and
after training, for several seeds. Saved to results/dog_game_trajectories.json,
drawn by doggame.dog_game_report and animated on the live site. Run with `python3 -m doggame.dog_game_trajectories`."""

import json
import random

import numpy as np
import torch

from doggame.dog_game_study import WALK_HOUSES
from doggame.env import Square
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.polar_agent import PolarPolicy, train_polar
from doggame.polar_dqn import ANGLES, NashQNetwork, choose_directions, train_nash_dqn

STEPS = 15
OUTPUT = "results/dog_game_trajectories.json"


def record(env, choose_moves, steps=STEPS):
    """Play one game. `choose_moves(state)` returns each player's (theta, r).
    Returns the positions of red, blue and the dog, from the start through every step."""
    state = env.reset()
    path = {"red": [env.red.tolist()], "blue": [env.blue.tolist()], "dog": [env.dog.tolist()]}
    for _ in range(steps):
        env.step(*choose_moves(state))
        state = env.state()
        for name in path:
            path[name].append(getattr(env, name).tolist())
    return path


def play_seed(env, seed):
    """Train both methods from this seed and record one game before and after training."""

    def seeded():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reinforce_moves(policy):
        return lambda state: tuple((theta, frac * env.max_step) for theta, frac in policy.act(state))

    def nash_q_moves(net):
        return lambda state: tuple((ANGLES[a], env.max_step) for a in choose_directions(net, state, 0.0))

    seeded()
    policy = PolarPolicy(seed=seed)
    reinforce = {"before": record(env, reinforce_moves(policy))}
    train_polar(env, policy, episodes=300, seed=seed)
    reinforce["after"] = record(env, reinforce_moves(policy))

    seeded()
    net = NashQNetwork()
    nash_q = {"before": record(env, nash_q_moves(net))}
    net = train_nash_dqn(env, episodes=60, seed=seed)
    nash_q["after"] = record(env, nash_q_moves(net))

    return dict(seed=seed, reinforce=reinforce, nash_q=nash_q)


def trajectories(seeds=range(5), path=OUTPUT):
    """Record every seed and save after each one, so a stopped run keeps what it finished."""
    env = PolarDogGameEnv(*WALK_HOUSES, w=0.5, max_step=0.2)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, Square())
    result = dict(house_red=env.house_red.tolist(), house_blue=env.house_blue.tolist(),
                  nash_red=nash_red.tolist(), nash_blue=nash_blue.tolist(), seeds=[])
    for seed in seeds:
        result["seeds"].append(play_seed(env, seed))
        with open(path, "w") as f:
            json.dump(result, f)
        print(f"seed {seed} recorded", flush=True)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    trajectories()
