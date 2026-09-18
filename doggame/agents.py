"""Gaussian policy agents built on PyTorch autograd.

A single joint module produces both players' action means, via one of
the doggame.networks architectures, plus a learned per-player log-std.
Training goes through torch.optim rather than a hand-derived gradient:
the meeting was explicit that once a network-training package is in
play, no one should be hand-writing gradient-descent math.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from doggame.networks import make_policy_network


class TwoPlayerPolicy(nn.Module):
    def __init__(self, domain, architecture="separate", seed=None, **network_kwargs):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.domain = domain
        self.network = make_policy_network(architecture, **network_kwargs)
        self.log_std_red = nn.Parameter(torch.full((2,), -0.5))
        self.log_std_blue = nn.Parameter(torch.full((2,), -0.5))

    def means(self, state):
        raw_red, raw_blue = self.network(state)
        span = self.domain.high - self.domain.low
        mean_red = self.domain.low + torch.sigmoid(raw_red) * span
        mean_blue = self.domain.low + torch.sigmoid(raw_blue) * span
        return mean_red, mean_blue

    def act(self, state):
        """Sample one action per player for a single state.

        Returns numpy actions, clipped to the domain and ready for
        env.step.
        """
        with torch.no_grad():
            state_t = torch.as_tensor(
                np.asarray(state, dtype=np.float32)
            ).unsqueeze(0)
            mean_red, mean_blue = self.means(state_t)
            action_red = Normal(mean_red, torch.exp(self.log_std_red)).sample()
            action_blue = Normal(mean_blue, torch.exp(self.log_std_blue)).sample()
        return (
            self.domain.clip(action_red.squeeze(0).numpy()),
            self.domain.clip(action_blue.squeeze(0).numpy()),
        )

    def log_prob(self, states, actions_red, actions_blue):
        """Batched log-probabilities, for a REINFORCE loss."""
        mean_red, mean_blue = self.means(states)
        logp_red = Normal(mean_red, torch.exp(self.log_std_red)).log_prob(
            actions_red
        ).sum(-1)
        logp_blue = Normal(mean_blue, torch.exp(self.log_std_blue)).log_prob(
            actions_blue
        ).sum(-1)
        return logp_red, logp_blue
