"""A minimal REINFORCE agent with a linear-Gaussian policy.

The action mean is an affine function of the state, squashed into the
domain with a logistic map; the log-std is a learned per-dimension
parameter. This is intentionally simple -- enough to show
policy-gradient self-play move toward the analytical stage-game Nash
equilibrium (see doggame.nash) without pulling in a deep-learning
dependency.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LinearGaussianPolicy:
    def __init__(self, domain, seed=None):
        self.domain = domain
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.1, size=(2, 2))
        self.b = rng.normal(scale=0.1, size=2)
        self.log_std = np.full(2, -0.5)

    def mean(self, state):
        raw = self.W @ np.asarray(state, dtype=float) + self.b
        unit = sigmoid(raw)
        span = self.domain.high - self.domain.low
        return self.domain.low + unit * span

    def act(self, state, rng):
        mean = self.mean(state)
        std = np.exp(self.log_std)
        action = mean + std * rng.normal(size=2)
        return self.domain.clip(action), mean, std

    def apply_gradients(self, grad_W, grad_b, grad_log_std, lr):
        self.W += lr * grad_W
        self.b += lr * grad_b
        self.log_std += lr * grad_log_std


def reinforce_update(policy, states, actions, means, stds, returns, lr, baseline=0.0):
    """One REINFORCE update from a batch of (state, action, mean, std,
    return) tuples collected under the current policy."""
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    advantages = np.asarray(returns, dtype=float) - baseline

    span = policy.domain.high - policy.domain.low
    grad_W = np.zeros_like(policy.W)
    grad_b = np.zeros_like(policy.b)
    grad_log_std = np.zeros_like(policy.log_std)

    for state, action, mean, std, advantage in zip(
        states, actions, means, stds, advantages
    ):
        raw = policy.W @ state + policy.b
        unit = sigmoid(raw)
        dmean_draw = span * unit * (1 - unit)

        dlogp_dmean = (action - mean) / std**2
        dlogp_draw = dlogp_dmean * dmean_draw
        dlogp_dlogstd = (action - mean) ** 2 / std**2 - 1

        grad_W += advantage * np.outer(dlogp_draw, state)
        grad_b += advantage * dlogp_draw
        grad_log_std += advantage * dlogp_dlogstd

    n = len(states)
    policy.apply_gradients(grad_W / n, grad_b / n, grad_log_std / n, lr)
