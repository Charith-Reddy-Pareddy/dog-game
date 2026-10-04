"""REINFORCE for the angle-radius dog game.

Each player's network reads the state (dog, red, blue positions) and
outputs a distribution over a move: the angle comes from a von Mises
(so 0 and 2*pi are the same direction, no wraparound problem) and the
step length from a Beta on [0, 1], scaled by the environment's
`max_step`. Both have exact log-probabilities, so nothing is clipped
before the gradient is taken.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta, VonMises

from doggame.train import discounted_returns

STATE_DIM = 6


def _player_net(hidden):
    # outputs: angle direction (2), angle concentration, radius alpha, radius beta
    return nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 5))


class PolarPolicy(nn.Module):
    def __init__(self, hidden=16, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.red = _player_net(hidden)
        self.blue = _player_net(hidden)

    @staticmethod
    def _distributions(net, state):
        out = net(state)
        angle = VonMises(torch.atan2(out[..., 1], out[..., 0]), F.softplus(out[..., 2]) + 1.0)
        radius = Beta(F.softplus(out[..., 3]) + 1.0, F.softplus(out[..., 4]) + 1.0)
        return angle, radius

    def act(self, state):
        """One sampled move per player: ((theta, r), (theta, r)), r in [0, 1]."""
        state = torch.as_tensor(state, dtype=torch.float32)
        moves = []
        with torch.no_grad():
            for net in (self.red, self.blue):
                angle, radius = self._distributions(net, state)
                moves.append((angle.sample().item(), radius.sample().item()))
        return moves[0], moves[1]

    def log_prob(self, states, moves_red, moves_blue):
        """Log-probability of each player's moves, for a batch of states."""
        result = []
        for net, moves in ((self.red, moves_red), (self.blue, moves_blue)):
            angle, radius = self._distributions(net, states)
            r = moves[:, 1].clamp(1e-4, 1 - 1e-4)
            result.append(angle.log_prob(moves[:, 0]) + radius.log_prob(r))
        return result[0], result[1]


def run_episode(env, policy, steps):
    state = env.reset()
    states, moves_red, moves_blue, rewards_red, rewards_blue = [], [], [], [], []
    for _ in range(steps):
        move_red, move_blue = policy.act(state)
        states.append(state)
        moves_red.append(move_red)
        moves_blue.append(move_blue)
        scaled = lambda m: (m[0], m[1] * env.max_step)
        state, (reward_red, reward_blue) = env.step(scaled(move_red), scaled(move_blue))
        rewards_red.append(reward_red)
        rewards_blue.append(reward_blue)
    return states, moves_red, moves_blue, rewards_red, rewards_blue


def train_polar(env, policy, episodes=300, steps=15, lr=0.01, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for _ in range(episodes):
        states, moves_red, moves_blue, rewards_red, rewards_blue = run_episode(env, policy, steps)

        advantages = []
        for rewards in (rewards_red, rewards_blue):
            returns = discounted_returns(rewards, env.discount)
            advantages.append(torch.as_tensor(returns - returns.mean(), dtype=torch.float32))

        as_tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)
        logp_red, logp_blue = policy.log_prob(as_tensor(states), as_tensor(moves_red), as_tensor(moves_blue))
        loss = -(logp_red * advantages[0] + logp_blue * advantages[1]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def final_positions(env, policy, steps=15):
    """Where each player ends up after one sampled episode."""
    run_episode(env, policy, steps)
    return env.red.copy(), env.blue.copy()
