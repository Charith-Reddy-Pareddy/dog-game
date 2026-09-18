"""Convergence study: does policy-gradient self-play actually reach
the analytical Nash equilibrium on the dog game?

That was the open question the meeting posed for the policy-gradient
subgroup ("can this converge? that's the big question"). One training
run isn't an answer; this runs several house/weight configurations
and architectures and reports, for each, the learned policy's
exploitability (doggame.verify) -- how much either player could still
gain by unilaterally deviating.
"""

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.env import DogGameEnv
from doggame.nash import solve_stage_nash
from doggame.train import train_self_play
from doggame.verify import exploitability

DEFAULT_CONFIGS = [
    dict(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5),
    dict(house_red=(0.8, 0.8), house_blue=(0.2, 0.3), w=0.3),
    dict(house_red=(0.7, 0.1), house_blue=(0.6, 0.9), w=0.7),
    dict(house_red=(0.95, 0.5), house_blue=(0.05, 0.5), w=0.5),
    dict(house_red=(0.6, 0.6), house_blue=(0.4, 0.4), w=0.5),
]


def _learned_means(policy, state):
    with torch.no_grad():
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        mean_red, mean_blue = policy.means(state_t)
    return mean_red.squeeze(0).numpy(), mean_blue.squeeze(0).numpy()


def run_convergence_study(
    configs=DEFAULT_CONFIGS,
    architecture="separate",
    episodes=1500,
    lr=0.01,
    seed=0,
):
    """Train self-play from scratch on each config and report how
    close the result gets to the analytical Nash equilibrium."""
    results = []
    for config in configs:
        env = DogGameEnv(**config, discount=0.9)
        policy = TwoPlayerPolicy(env.domain, architecture=architecture, seed=seed)
        train_self_play(env, policy, episodes=episodes, lr=lr, seed=seed)

        state = env.reset()
        learned_red, learned_blue = _learned_means(policy, state)

        nash_red, nash_blue, _ = solve_stage_nash(
            env.house_red, env.house_blue, env.w, env.domain
        )
        red_gap, blue_gap = exploitability(
            learned_red, learned_blue, env.house_red, env.house_blue, env.w, env.domain
        )

        results.append(
            dict(
                config=config,
                distance_red=float(np.linalg.norm(learned_red - nash_red)),
                distance_blue=float(np.linalg.norm(learned_blue - nash_blue)),
                red_exploitability=red_gap,
                blue_exploitability=blue_gap,
            )
        )
    return results


if __name__ == "__main__":
    for architecture in ("separate", "shared", "partial"):
        print(f"\n=== architecture: {architecture} ===")
        for result in run_convergence_study(architecture=architecture):
            print(result["config"])
            print(
                f"  distance to Nash:  red={result['distance_red']:.4f}"
                f"  blue={result['distance_blue']:.4f}"
            )
            print(
                f"  exploitability:    red={result['red_exploitability']:.5f}"
                f"  blue={result['blue_exploitability']:.5f}"
            )
