"""Exploratory cross-house robustness follow-up for polar actions."""

import json

import numpy as np
import torch

from doggame.agents import BoundedCartesianPolicy, PolarActionPolicy
from doggame.env import DogGameEnv
from doggame.experiments import DEFAULT_CONFIGS
from doggame.polar_action_study import CHECKPOINTS, EPISODES, STEPS, _read, _score
from doggame.train import train_self_play

OUTPUT = "results/polar_action_pg_robustness.jsonl"
REPRESENTATIONS = ("cartesian_beta", "polar_beta")
CONFIGS = (1, 2, 3, 4)


def run(seeds=range(10), path=OUTPUT):
    """Run fixed REINFORCE comparisons; save each completed run once."""
    rows = _read(path)
    done = {
        (r["key"]["representation"], r["key"]["seed"], r["key"]["config"])
        for r in rows if r.get("study") == "polar_action_pg_robustness"
    }
    torch.set_num_threads(1)
    for config_id in CONFIGS:
        config = DEFAULT_CONFIGS[config_id]
        for representation in REPRESENTATIONS:
            for seed in seeds:
                key = (representation, int(seed), config_id)
                if key in done:
                    continue
                env = DogGameEnv(**config, discount=0.9)
                if representation == "polar_beta":
                    policy = PolarActionPolicy(env.domain, architecture="separate", seed=int(seed))
                else:
                    policy = BoundedCartesianPolicy(env.domain, architecture="separate", seed=int(seed))
                curve = {}

                def record(episode):
                    if episode in CHECKPOINTS:
                        curve[str(episode)] = _score(
                            policy, env, representation, seed=100000 + int(seed)
                        )

                train_self_play(
                    env, policy, episodes=EPISODES, steps_per_episode=STEPS,
                    lr=0.01, seed=int(seed), callback=record,
                )
                row = {
                    "study": "polar_action_pg_robustness",
                    "protocol": {
                        "episodes": EPISODES, "steps": STEPS,
                        "checkpoints": CHECKPOINTS, "eval_samples": 8192,
                        "discount": 0.9, "algorithm": "reinforce",
                        "protocol_version": 1,
                    },
                    "key": {
                        "representation": representation, "algorithm": "reinforce",
                        "seed": int(seed), "config": config_id,
                    },
                    "curve": curve,
                }
                with open(path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                done.add(key)
                print(f"saved config {config_id} {representation} seed {seed}", flush=True)
    return _read(path)


def summary(path=OUTPUT):
    from scipy.stats import t

    rows = [r for r in _read(path) if r.get("study") == "polar_action_pg_robustness"]
    for config_id in CONFIGS:
        by_rep = {
            rep: {
                r["key"]["seed"]: r["curve"][str(EPISODES)]["expected_exploitability"]
                for r in rows
                if r["key"]["config"] == config_id and r["key"]["representation"] == rep
            }
            for rep in REPRESENTATIONS
        }
        shared = sorted(set(by_rep["polar_beta"]) & set(by_rep["cartesian_beta"]))
        if not shared:
            continue
        delta = np.array([by_rep["polar_beta"][s] - by_rep["cartesian_beta"][s] for s in shared])
        if len(delta) > 1:
            margin = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
        else:
            margin = float("nan")
        print(
            f"config={config_id} n={len(delta)} polar-cartesian-beta mean={delta.mean():.6f} "
            f"95% t CI=[{delta.mean()-margin:.6f}, {delta.mean()+margin:.6f}]"
        )


if __name__ == "__main__":
    run()
