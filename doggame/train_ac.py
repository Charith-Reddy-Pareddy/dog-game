"""Actor-critic training for the dog game: A2C and PPO.

Same setting as doggame.train (one self-play episode per update, a Gaussian
policy for both players), but each player also learns a value function of the
dog's position, and the policy gradient uses `return - value` as the advantage
instead of `return - mean return`. PPO additionally reuses each episode for a
few epochs, with the probability ratio clipped so a step cannot move the
policy too far.
"""

import numpy as np
import torch
from torch import nn

from doggame.train import discounted_returns, run_episode

ALGORITHMS = ("a2c", "ppo")


def _value_net(state_dim, hidden=32):
    return nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def train_actor_critic(env, policy, algorithm="a2c", episodes=500, steps_per_episode=20, lr=0.01, seed=0,
                       callback=None, ppo_epochs=4, clip=0.2, rollout=run_episode, state_dim=2, entropy=0.0):
    """Train `policy` by self-play with A2C or PPO. `callback(episodes_done)`, if
    given, is called after every update. `rollout` plays one episode and `state_dim` is
    the size of its states, so the same loop trains the walking game's policy too
    (see doggame.polar_agent). With `entropy` > 0, each player's loss also rewards its
    policy's entropy by that coefficient (the policy needs an `entropy(states)` method).
    Returns the two players' value networks."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    torch.manual_seed(seed)
    values = nn.ModuleList([_value_net(state_dim), _value_net(state_dim)])
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(values.parameters()), lr=lr)
    as_tensor = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)

    for episode in range(episodes):
        states, actions_red, actions_blue, rewards_red, rewards_blue = rollout(env, policy, steps_per_episode)
        states_t, actions_red_t, actions_blue_t = as_tensor(states), as_tensor(actions_red), as_tensor(actions_blue)
        returns = [as_tensor(discounted_returns(r, env.discount)) for r in (rewards_red, rewards_blue)]
        with torch.no_grad():
            old = policy.log_prob(states_t, actions_red_t, actions_blue_t)

        for _ in range(ppo_epochs if algorithm == "ppo" else 1):
            logps = policy.log_prob(states_t, actions_red_t, actions_blue_t)
            loss = 0.0
            entropies = policy.entropy(states_t) if entropy else (0.0, 0.0)
            for logp, old_logp, value_net, ret, ent in zip(logps, old, values, returns, entropies):
                value = value_net(states_t).squeeze(1)
                advantage = (ret - value).detach()
                if algorithm == "ppo":
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    ratio = (logp - old_logp).exp()
                    gain = torch.min(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
                else:
                    gain = (logp * advantage).mean()
                loss = loss - gain + 0.5 * nn.functional.mse_loss(value, ret) - entropy * torch.as_tensor(ent).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if callback:
            callback(episode + 1)
    return values
