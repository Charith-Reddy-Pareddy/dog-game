"""Experiments that answer open research questions about the dog game.

Each study is a generator of (key, run) pairs; `run()` executes the ones missing
from the results file and appends each finished row at once, so a stopped run
keeps what it did and a rerun skips it. Run everything with
`python3 -m doggame.dog_game_studies`, or pick studies by name; add a last argument like `1/4`
to do only the second of four shards.

- algorithms:   REINFORCE vs A2C vs PPO, how fast each gets near the equilibrium
- walking:      the same three algorithms on the angle-radius (walking) game
- selection:    which equilibrium self-play finds when the houses share a coordinate
- inertia:      does giving the dog momentum (w0) change where training ends up
- pretraining:  does starting from a policy trained on another setup help
- stage_solver: how precisely the Nash-Q stage game has to be solved
- resolution:   how the action-grid resolution affects the discretized solution
- noise:        how much of fictitious play's mixing is just noise
"""

import json
import os
import sys
import time

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.dog_game_study import WALK_HOUSES, walk_distance
from doggame.discretize import action_grid, payoff_matrices
from doggame.dqn import solve_discretized_nash_q
from doggame.env import DogGameEnv, Square, make_inertial_transition, negative_squared_distance_reward, weighted_average_transition
from doggame.exact_nash import lemke_howson_nash
from doggame.experiments import DEFAULT_CONFIGS
from doggame.fictitious_play import expected_value, fictitious_play
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.polar_agent import PolarPolicy, final_positions, train_polar, train_polar_actor_critic
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic
from doggame.verify import exploitability, mixed_strategy_indifference_gap

RESULTS = "results/dog_game_studies.jsonl"
EPISODES = 1500
CONVERGED = 0.004  # exploitability below this counts as converged
DEGENERATE = 3  # the configuration whose houses share a y-coordinate
SLOW = 1  # the configuration where one run needed more than 1500 episodes
STAGE_CONFIG = DEFAULT_CONFIGS[0]


def make_env(config, w0=0.0):
    return DogGameEnv(**config, discount=0.9, transition_fn=make_inertial_transition(w0))


def learned_actions(policy, env):
    """The policy's mean action for each player at the starting state."""
    with torch.no_grad():
        mean_red, mean_blue = policy.means(torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0))
    return mean_red.squeeze(0).numpy(), mean_blue.squeeze(0).numpy()


def score(policy, env):
    """How far the learned policy is from the stage-game equilibrium, and how much either player could gain."""
    red, blue = learned_actions(policy, env)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    gap = max(exploitability(red, blue, env.house_red, env.house_blue, env.w, env.domain))
    return dict(exploitability=float(gap), distance=float(np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)),
                red=red.tolist(), blue=blue.tolist())


def train_with_curve(env, policy, algorithm, episodes, checkpoints, seed):
    """Train, scoring the policy at each checkpoint. Returns {episodes so far: exploitability}."""
    curve = {}

    def record(done):
        if done in checkpoints:
            curve[done] = score(policy, env)["exploitability"]

    if algorithm == "reinforce":
        train_self_play(env, policy, episodes=episodes, seed=seed, callback=record)
    else:
        train_actor_critic(env, policy, algorithm, episodes=episodes, seed=seed, callback=record)
    return curve


def study_algorithms(seeds):
    checkpoints = (100, 250, 500, 1000, 1500)
    for config_index, config in enumerate(DEFAULT_CONFIGS):
        for algorithm in ("reinforce", "a2c", "ppo"):
            for seed in seeds:
                def run(config=config, algorithm=algorithm, seed=seed):
                    env = make_env(config)
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
                    return dict(curve=train_with_curve(env, policy, algorithm, EPISODES, checkpoints, seed))
                yield dict(config=config_index, algorithm=algorithm, seed=seed), run


def study_walking(seeds):
    """REINFORCE, A2C and PPO on the walking game: how far the players end from the one-shot Nash corners."""
    checkpoints = tuple(range(25, 301, 25))  # finely, because the walking policies' paths swing
    for algorithm in ("reinforce", "a2c", "ppo"):
        for seed in seeds:
            def run(algorithm=algorithm, seed=seed):
                env = PolarDogGameEnv(*WALK_HOUSES, w=0.5, max_step=0.2)
                torch.manual_seed(seed)
                policy = PolarPolicy(seed=seed)
                distance = lambda: walk_distance(env, lambda: final_positions(env, policy))
                start, curve = distance(), {}

                def record(done):
                    if done in checkpoints:
                        curve[done] = distance()

                if algorithm == "reinforce":
                    train_polar(env, policy, episodes=300, seed=seed, callback=record)
                else:
                    train_polar_actor_critic(env, policy, algorithm, episodes=300, seed=seed, callback=record)
                return dict(start=start, curve=curve)
            yield dict(algorithm=algorithm, seed=seed), run


def study_selection(seeds):
    for architecture in ("separate", "shared", "partial"):
        for seed in range(10):  # equilibrium selection is a question about frequencies, so use more seeds
            def run(architecture=architecture, seed=seed):
                env = make_env(DEFAULT_CONFIGS[DEGENERATE])
                policy = TwoPlayerPolicy(env.domain, architecture=architecture, seed=seed)
                train_self_play(env, policy, episodes=EPISODES, seed=seed)
                return score(policy, env)
            yield dict(architecture=architecture, seed=seed), run


def study_inertia(seeds):
    for config_index in (0, SLOW):
        for w0 in (0.0, 0.3, 0.6, 0.9):
            for seed in seeds:
                def run(config_index=config_index, w0=w0, seed=seed):
                    env = make_env(DEFAULT_CONFIGS[config_index], w0)
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
                    train_self_play(env, policy, episodes=EPISODES, seed=seed)
                    return score(policy, env)
                yield dict(config=config_index, w0=w0, seed=seed), run


def study_pretraining(seeds):
    checkpoints = (100, 300)
    for source in (None, 0, 4):
        for seed in seeds:
            def run(source=source, seed=seed):
                policy = TwoPlayerPolicy(make_env(DEFAULT_CONFIGS[SLOW]).domain, architecture="separate", seed=seed)
                if source is not None:
                    train_self_play(make_env(DEFAULT_CONFIGS[source]), policy, episodes=EPISODES, seed=seed)
                env = make_env(DEFAULT_CONFIGS[SLOW])
                return dict(curve=train_with_curve(env, policy, "reinforce", 300, checkpoints, seed))
            yield dict(source=source, seed=seed), run


def stage_game(resolution, config=STAGE_CONFIG):
    domain = Square(0.0, 1.0)
    grid = action_grid(domain, resolution)
    payoff_red, payoff_blue = payoff_matrices(grid, config["house_red"], config["house_blue"], config["w"], domain,
                                              weighted_average_transition, negative_squared_distance_reward)
    return grid, payoff_red, payoff_blue


def nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue):
    """The most either player could gain by switching to a single best action."""
    value_red = expected_value(payoff_red, strategy_red, strategy_blue)
    value_blue = expected_value(payoff_blue, strategy_red, strategy_blue)
    return float(max((payoff_red @ strategy_blue).max() - value_red, (strategy_red @ payoff_blue).max() - value_blue))


def study_stage_solver(seeds, resolution=9):
    _, payoff_red, payoff_blue = stage_game(resolution)
    exact_q = solve_discretized_nash_q(payoff_red, payoff_blue, 0.9, solver="exact")[0]
    for iterations in (10, 30, 100, 300, 1000, 3000):
        for resync_every in (1, 5, 25):
            for seed in seeds:
                def run(iterations=iterations, resync_every=resync_every, seed=seed):
                    start = time.time()
                    q_red, _, strategy_red, strategy_blue = solve_discretized_nash_q(
                        payoff_red, payoff_blue, 0.9, fictitious_play_iterations=iterations, resync_every=resync_every, seed=seed)
                    return dict(q_error=float(np.abs(q_red - exact_q).max()), gap=nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue),
                                seconds=time.time() - start)
                yield dict(iterations=iterations, resync_every=resync_every, seed=seed), run


def study_resolution(seeds):
    for config_index, config in enumerate(DEFAULT_CONFIGS):
        nash_red, nash_blue, _ = solve_stage_nash(config["house_red"], config["house_blue"], config["w"], Square(0.0, 1.0))
        for resolution in (3, 5, 9, 13, 17):
            def run(config=config, resolution=resolution, nash_red=nash_red, nash_blue=nash_blue):
                start = time.time()
                grid, payoff_red, payoff_blue = stage_game(resolution, config)
                strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)
                red, blue = grid[strategy_red.argmax()], grid[strategy_blue.argmax()]
                return dict(actions=len(grid), support=[int((strategy_red > 1e-6).sum()), int((strategy_blue > 1e-6).sum())],
                            distance=float(np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)), seconds=time.time() - start)
            yield dict(config=config_index, resolution=resolution), run


def study_noise(seeds, resolution=9):
    _, payoff_red, payoff_blue = stage_game(resolution)
    for iterations in (30, 100, 300, 1000, 3000, 10000):
        for seed in seeds:
            def run(iterations=iterations, seed=seed):
                strategy_red, strategy_blue = fictitious_play(payoff_red, payoff_blue, iterations=iterations, seed=seed)
                spread, gap = mixed_strategy_indifference_gap(payoff_red, strategy_red, strategy_blue)
                return dict(spread=float(spread), gap=float(gap), support=int((strategy_red > 1e-3).sum()), top=float(strategy_red.max()))
            yield dict(iterations=iterations, seed=seed), run
    exact_red, exact_blue = lemke_howson_nash(payoff_red, payoff_blue)
    spread, gap = mixed_strategy_indifference_gap(payoff_red, exact_red, exact_blue)
    yield dict(iterations="exact", seed=0), lambda: dict(spread=float(spread), gap=float(gap), support=int((exact_red > 1e-3).sum()), top=float(exact_red.max()))


STUDIES = dict(algorithms=study_algorithms, walking=study_walking, selection=study_selection, inertia=study_inertia, pretraining=study_pretraining,
               stage_solver=study_stage_solver, resolution=study_resolution, noise=study_noise)


def finished(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.dumps([r["study"], r["key"]], sort_keys=True) for r in map(json.loads, f)}


def run(names=tuple(STUDIES), seeds=range(5), path=RESULTS, shard=(0, 1)):
    """Run the named studies, skipping runs already in `path`, and append each result as it finishes.
    With shard=(i, n), only every n-th run (starting at i) is done, so n processes can split the work."""
    index, count = shard
    for name in names:
        for position, (key, execute) in enumerate(STUDIES[name](seeds)):
            if position % count != index or json.dumps([name, key], sort_keys=True) in finished(path):
                continue
            row = dict(study=name, key=key, **execute())
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(name, key, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs("results", exist_ok=True)
    args = sys.argv[1:]
    shard = tuple(int(x) for x in args.pop().split("/")) if args and "/" in args[-1] else (0, 1)
    run(args or list(STUDIES), shard=shard)
