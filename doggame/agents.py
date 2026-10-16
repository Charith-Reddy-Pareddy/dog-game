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
import torch.nn.functional as F
from torch.distributions import Beta, Normal, VonMises

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


class PolarActionPolicy(nn.Module):
    """Stochastic dog-game policy whose absolute point actions are (angle, radius).

    The polar origin is the center of the square. The radius is a fraction in
    [0, 1] of the distance from that center to the square boundary along the
    sampled ray. This maps every latent action to a feasible point without
    clipping, while retaining an invertible mapping almost everywhere.
    """

    def __init__(self, domain, architecture="separate", seed=None, **network_kwargs):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.domain = domain
        self.center = (float(domain.low) + float(domain.high)) / 2
        self.half_width = (float(domain.high) - float(domain.low)) / 2
        if self.half_width <= 0:
            raise ValueError("domain must have positive width")
        self.network = make_policy_network(architecture, action_dim=5, **network_kwargs)

    @staticmethod
    def _distribution(output):
        angle = VonMises(torch.atan2(output[..., 1], output[..., 0]), F.softplus(output[..., 2]) + 1.0)
        radius = Beta(F.softplus(output[..., 3]) + 1.0, F.softplus(output[..., 4]) + 1.0)
        return angle, radius

    def _to_action(self, theta, fraction):
        dx, dy = torch.cos(theta), torch.sin(theta)
        ray_norm = torch.maximum(dx.abs(), dy.abs()).clamp_min(1e-8)
        radius = self.half_width * fraction / ray_norm
        return torch.stack((self.center + radius * dx, self.center + radius * dy), dim=-1)

    def _from_action(self, action):
        delta = action - self.center
        theta = torch.atan2(delta[..., 1], delta[..., 0])
        fraction = delta.abs().amax(dim=-1) / self.half_width
        return theta, fraction

    def _distributions(self, state):
        red, blue = self.network(state)
        return self._distribution(red), self._distribution(blue)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            (angle_r, radius_r), (angle_b, radius_b) = self._distributions(state)
            red = self._to_action(angle_r.sample(), radius_r.sample())
            blue = self._to_action(angle_b.sample(), radius_b.sample())
        return red.squeeze(0).numpy(), blue.squeeze(0).numpy()

    def sample_action_batch(self, states, samples=8192, seed=0):
        """Draw reproducible action samples shaped (samples, batch, 2)."""
        state = torch.as_tensor(states, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                (angle_r, radius_r), (angle_b, radius_b) = self._distributions(state)
                red = self._to_action(angle_r.sample((samples,)), radius_r.sample((samples,)))
                blue = self._to_action(angle_b.sample((samples,)), radius_b.sample((samples,)))
        return red, blue

    def means(self, state):
        """Monte Carlo expected Cartesian actions, for compatibility with scorers."""
        state = torch.as_tensor(state, dtype=torch.float32)
        red, blue = self.sample_action_batch(state, samples=8192, seed=1729)
        return red.mean(dim=0), blue.mean(dim=0)

    def log_prob(self, states, actions_red, actions_blue):
        """Log-probabilities of feasible Cartesian points under the latent polar policy."""
        result = []
        for (angle, radius), actions in zip(
            self._distributions(states), (actions_red, actions_blue)
        ):
            theta, fraction = self._from_action(actions)
            fraction = fraction.clamp(1e-6, 1 - 1e-6)
            result.append(angle.log_prob(theta) + radius.log_prob(fraction))
        return result[0], result[1]


class BoundedCartesianPolicy(nn.Module):
    """Policy with independent Beta coordinates, a bounded non-polar control."""

    def __init__(self, domain, architecture="separate", seed=None, **network_kwargs):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.domain = domain
        self.span = float(domain.high) - float(domain.low)
        if self.span <= 0:
            raise ValueError("domain must have positive width")
        self.network = make_policy_network(architecture, action_dim=4, **network_kwargs)

    @staticmethod
    def _distribution(output):
        shape = F.softplus(output.reshape(*output.shape[:-1], 2, 2)) + 1.0
        return Beta(shape[..., 0], shape[..., 1])

    def _distributions(self, state):
        red, blue = self.network(state)
        return self._distribution(red), self._distribution(blue)

    def _to_action(self, fraction):
        return self.domain.low + self.span * fraction

    def _from_action(self, action):
        return (action - self.domain.low) / self.span

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            red, blue = self._distributions(state)
            return self._to_action(red.sample()).squeeze(0).numpy(), self._to_action(blue.sample()).squeeze(0).numpy()

    def sample_action_batch(self, states, samples=8192, seed=0):
        state = torch.as_tensor(states, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                red, blue = self._distributions(state)
                return self._to_action(red.sample((samples,))), self._to_action(blue.sample((samples,)))

    def means(self, state):
        red, blue = self._distributions(torch.as_tensor(state, dtype=torch.float32))
        return self._to_action(red.mean), self._to_action(blue.mean)

    def log_prob(self, states, actions_red, actions_blue):
        result = []
        for distribution, actions in zip(self._distributions(states), (actions_red, actions_blue)):
            fraction = self._from_action(actions).clamp(1e-6, 1 - 1e-6)
            result.append(distribution.log_prob(fraction).sum(dim=-1) - 2 * np.log(self.span))
        return result[0], result[1]
