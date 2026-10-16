"""Paired Cartesian-versus-polar policy-gradient study on the original dog game.

The polar actor emits an angle and a radial fraction from the square's center;
the ray is mapped exactly to the square boundary. DQN is not used here.
"""

import json
import os
import sys

import numpy as np
import torch

from doggame.agents import BoundedCartesianPolicy, PolarActionPolicy, TwoPlayerPolicy
from doggame.env import DogGameEnv
from doggame.experiments import DEFAULT_CONFIGS
from doggame.nash import solve_stage_nash
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic

OUTPUT = "results/polar_action_pg.jsonl"
PRIMARY_CONFIG = 0
ALGORITHMS = ("reinforce", "a2c", "ppo")
REPRESENTATIONS = ("cartesian_gaussian_clipped", "cartesian_beta", "polar_beta")
EPISODES = 1500
STEPS = 20
CHECKPOINTS = (100, 250, 500, 1000, 1500)
EVAL_SAMPLES = 8192
THRESHOLD = 0.004


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample_cartesian(policy, state, samples, seed):
    state = torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, -1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            red_mean, blue_mean = policy.means(state)
            red = torch.distributions.Normal(red_mean, torch.exp(policy.log_std_red)).sample((samples,))
            blue = torch.distributions.Normal(blue_mean, torch.exp(policy.log_std_blue)).sample((samples,))
            red = red.clamp(policy.domain.low, policy.domain.high)
            blue = blue.clamp(policy.domain.low, policy.domain.high)
    return red[:, 0].cpu().numpy(), blue[:, 0].cpu().numpy()


def action_moments(policy, state, representation, samples=EVAL_SAMPLES, seed=99173):
    """Estimate means, covariance traces, and clipped mass for independent actions."""
    if representation in ("polar_beta", "cartesian_beta"):
        red, blue = policy.sample_action_batch(state, samples=samples, seed=seed)
        red, blue = red[:, 0].cpu().numpy(), blue[:, 0].cpu().numpy()
    else:
        red, blue = _sample_cartesian(policy, state, samples, seed)
    return {
        "red_mean": red.mean(axis=0),
        "blue_mean": blue.mean(axis=0),
        "red_var": float(np.var(red, axis=0, ddof=1).sum()),
        "blue_var": float(np.var(blue, axis=0, ddof=1).sum()),
        "red_boundary_mass": float(np.mean(((red == policy.domain.low) | (red == policy.domain.high)).any(axis=1))),
        "blue_boundary_mass": float(np.mean(((blue == policy.domain.low) | (blue == policy.domain.high)).any(axis=1))),
    }


def expected_profile_gap(moments, env):
    """Exact best-response gains for the estimated action moments.

    For quadratic distance rewards, the expectation depends on each action
    distribution only through its mean and trace covariance. The best response
    to the opponent's distribution is the projection of the desired action
    onto the square. Monte Carlo is used only to estimate these moments.
    """
    red, blue = moments["red_mean"], moments["blue_mean"]
    dog_mean = env.w * red + (1 - env.w) * blue
    red_best_action = env.domain.clip((env.house_red - (1 - env.w) * blue) / env.w)
    blue_best_action = env.domain.clip((env.house_blue - env.w * red) / (1 - env.w))

    red_current = -np.sum((dog_mean - env.house_red) ** 2) - env.w**2 * moments["red_var"] - (1 - env.w)**2 * moments["blue_var"]
    red_best = -np.sum((env.w * red_best_action + (1 - env.w) * blue - env.house_red) ** 2) - (1 - env.w)**2 * moments["blue_var"]
    blue_current = -np.sum((dog_mean - env.house_blue) ** 2) - env.w**2 * moments["red_var"] - (1 - env.w)**2 * moments["blue_var"]
    blue_best = -np.sum((env.w * red + (1 - env.w) * blue_best_action - env.house_blue) ** 2) - env.w**2 * moments["red_var"]
    return max(0.0, float(red_best - red_current)), max(0.0, float(blue_best - blue_current))


def _score(policy, env, representation, seed):
    state = env.reset()
    moments = action_moments(policy, state, representation, seed=seed)
    gaps = expected_profile_gap(moments, env)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    return {
        "expected_exploitability": float(max(gaps)),
        "red_gap": gaps[0], "blue_gap": gaps[1],
        "red_mean": moments["red_mean"].tolist(), "blue_mean": moments["blue_mean"].tolist(),
        "mean_action_distance": float(np.linalg.norm(moments["red_mean"] - nash_red) + np.linalg.norm(moments["blue_mean"] - nash_blue)),
        "red_var": moments["red_var"], "blue_var": moments["blue_var"],
        "red_boundary_mass": moments["red_boundary_mass"],
        "blue_boundary_mass": moments["blue_boundary_mass"],
    }


def _done(rows):
    return {(r["key"]["representation"], r["key"]["algorithm"], r["key"]["seed"], r["key"]["config"]) for r in rows}


def run(seeds=range(10), path=OUTPUT):
    """Run the fixed protocol; completed rows are appended and never overwritten."""
    rows = _read(path)
    done = _done(rows)
    config = DEFAULT_CONFIGS[PRIMARY_CONFIG]
    torch.set_num_threads(1)
    for representation in REPRESENTATIONS:
        for algorithm in ALGORITHMS:
            for seed in seeds:
                key = (representation, algorithm, int(seed), PRIMARY_CONFIG)
                if key in done:
                    continue
                env = DogGameEnv(**config, discount=0.9)
                if representation == "polar_beta":
                    policy = PolarActionPolicy(env.domain, architecture="separate", seed=int(seed))
                elif representation == "cartesian_beta":
                    policy = BoundedCartesianPolicy(env.domain, architecture="separate", seed=int(seed))
                else:
                    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=int(seed))
                curve = {}

                def record(episode):
                    if episode in CHECKPOINTS:
                        curve[str(episode)] = _score(policy, env, representation, seed=100000 + int(seed))

                if algorithm == "reinforce":
                    train_self_play(env, policy, episodes=EPISODES, steps_per_episode=STEPS, lr=0.01, seed=int(seed), callback=record)
                else:
                    train_actor_critic(env, policy, algorithm, episodes=EPISODES, steps_per_episode=STEPS,
                                       lr=0.01, seed=int(seed), callback=record, ppo_epochs=4, clip=0.2)
                row = {
                    "study": "polar_action_pg",
                    "protocol": {"episodes": EPISODES, "steps": STEPS, "checkpoints": CHECKPOINTS,
                                 "eval_samples": EVAL_SAMPLES, "threshold": THRESHOLD, "protocol_version": 1},
                    "key": {"representation": representation, "algorithm": algorithm, "seed": int(seed), "config": PRIMARY_CONFIG},
                    "curve": curve,
                }
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"saved {representation} {algorithm} seed {seed}", flush=True)
    return _read(path)


def summary(path=OUTPUT):
    rows = _read(path)
    print("representation algorithm n converged final expected-exploitability mean (SD) median; paired polar - Cartesian 95% t CI")
    from scipy.stats import t

    for algorithm in ALGORITHMS:
        by_rep = {}
        for representation in REPRESENTATIONS:
            mine = [r for r in rows if r["study"] == "polar_action_pg" and r["key"]["algorithm"] == algorithm and r["key"]["representation"] == representation]
            vals = np.array([r["curve"][str(EPISODES)]["expected_exploitability"] for r in mine], dtype=float)
            by_rep[representation] = {r["key"]["seed"]: r["curve"][str(EPISODES)]["expected_exploitability"] for r in mine}
            if len(vals):
                print(f"{representation:>10} {algorithm:>9} {len(vals):2d} {int((vals < THRESHOLD).sum())}/{len(vals)} "
                      f"{vals.mean():.6f} ({vals.std(ddof=1) if len(vals)>1 else 0:.6f}) {np.median(vals):.6f}")
        for comparator in ("cartesian_beta", "cartesian_gaussian_clipped"):
            shared = sorted(set(by_rep[comparator]) & set(by_rep["polar_beta"]))
            if len(shared) > 1:
                delta = np.array([by_rep["polar_beta"][s] - by_rep[comparator][s] for s in shared])
                half = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
                print(f"  paired polar_beta - {comparator} ({len(delta)} seeds): {delta.mean():.6f} [{delta.mean()-half:.6f}, {delta.mean()+half:.6f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(sys.argv[2] if len(sys.argv) > 2 else OUTPUT)
    else:
        chosen = range(int(sys.argv[1])) if len(sys.argv) > 1 else range(10)
        run(chosen)
