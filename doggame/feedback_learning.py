"""Does policy-gradient self-play learn a feedback equilibrium of the game with inertia?

Training starts each episode from a random dog position, so the learners see the states off the equilibrium path.
A learned policy is judged by its feedback exploitability: the most a player gains at a dog position by changing its
action for one round and then following the learned policy again (the one-shot deviation principle), with the
continuation simulated exactly. A feedback equilibrium scores 0, whichever equilibrium it is.
"""

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.env import DogGameEnv, make_inertial_transition
from doggame.inertia_theory import explicit_policy, nash_1d
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic


class RandomStart(DogGameEnv):
    """The dog game whose episodes start from a uniformly random dog position."""

    def reset(self):
        self.dog = self.domain.sample(self.rng)
        return self.dog.copy()


def learn(config, w0, algorithm, seed, episodes):
    env = RandomStart(**config, discount=0.9, transition_fn=make_inertial_transition(w0), seed=seed)
    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=seed)
    if algorithm == "reinforce":
        train_self_play(env, policy, episodes=episodes, seed=seed)
    else:
        train_actor_critic(env, policy, algorithm, episodes=episodes, seed=seed)
    return policy


def mean_policy(policy):
    """The learned policy's mean actions as a function of an (n, 2) array of dog positions."""
    def act(x):
        with torch.no_grad():
            red, blue = policy.means(torch.as_tensor(np.asarray(x), dtype=torch.float32))
        return red.numpy().astype(float), blue.numpy().astype(float)
    return act


def explicit_2d(config, w0, discount=0.9):
    """Proposition 2's policy in each coordinate, as a function of an (n, 2) array of dog positions."""
    parts = [explicit_policy(config["house_red"][c], config["house_blue"][c], config["w"],
                             *nash_1d(config["house_red"][c], config["house_blue"][c], config["w"])[:2], w0, discount) for c in range(2)]

    def act(x):
        x = np.asarray(x, dtype=float)
        pairs = [part(x[:, c]) for c, part in enumerate(parts)]
        return np.stack([p[0] for p in pairs], 1), np.stack([p[1] for p in pairs], 1)
    return act


def holding_2d(config):
    """Hold the one-shot Nash actions at every dog position."""
    actions = [nash_1d(config["house_red"][c], config["house_blue"][c], config["w"])[:2] for c in range(2)]
    red, blue = np.array([a[0] for a in actions]), np.array([a[1] for a in actions])
    return lambda x: (np.tile(red, (len(x), 1)), np.tile(blue, (len(x), 1)))


def feedback_gain(act, config, w0, discount=0.9, states=11, grid=21, rounds=120):
    """The largest and the mean (over dog positions and players) one-round deviation gain against `act`."""
    w, houses = config["w"], (np.asarray(config["house_red"], float), np.asarray(config["house_blue"], float))
    axis = np.linspace(0, 1, grid)
    deviations = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    gains = []
    for x in np.stack(np.meshgrid(np.linspace(0, 1, states), np.linspace(0, 1, states), indexing="ij"), -1).reshape(-1, 2):
        red, blue = (a[0] for a in act(x[None]))
        for i in range(2):
            own = np.vstack([deviations, red if i == 0 else blue])  # every deviation, and the policy's own action last
            a_red, a_blue = (own, np.tile(blue, (len(own), 1))) if i == 0 else (np.tile(red, (len(own), 1)), own)
            z = w0 * x + (1 - w0) * (w * a_red + (1 - w) * a_blue)
            total = -np.sum((z - houses[i]) ** 2, 1)
            for k in range(1, rounds):
                r, b = act(z)
                z = w0 * z + (1 - w0) * (w * r + (1 - w) * b)
                total -= discount ** k * np.sum((z - houses[i]) ** 2, 1)
            gains.append(max(total[:-1].max() - total[-1], 0.0))
    return float(np.max(gains)), float(np.mean(gains))
