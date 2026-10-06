"""Deep Nash-Q for the angle-radius dog game, with 10 compass directions.

Each player can walk in one of `N_DIRECTIONS` evenly spaced directions
(a full `max_step` each time). One network reads the state (dog, red,
blue) and outputs both players' Q-values for every pair of directions,
Q(s, a_red, a_blue). The learning target is the usual Nash-Q one,
`r + discount * NashValue(Q_target(s'))`, where the Nash equilibrium of
the 10x10 stage game comes from the exact Lemke-Howson solver. A frozen
target network and a replay buffer keep the training stable.
"""

import random

import numpy as np
import torch
from torch import nn

from doggame.exact_nash import lemke_howson_nash

N_DIRECTIONS = 10
ANGLES = 2 * np.pi * np.arange(N_DIRECTIONS) / N_DIRECTIONS


class NashQNetwork(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * N_DIRECTIONS**2),
        )

    def forward(self, states):
        """Returns (q_red, q_blue), each shaped (batch, 10, 10)."""
        out = self.body(states).view(-1, 2, N_DIRECTIONS, N_DIRECTIONS)
        return out[:, 0], out[:, 1]


def nash_strategies(q_red, q_blue):
    """The stage-game equilibrium of one state's pair of 10x10 Q matrices."""
    return lemke_howson_nash(q_red, q_blue, epsilon=1e-9)


def nash_values(q_red, q_blue):
    """Each player's value under the equilibrium, for a batch of states."""
    values = []
    for qr, qb in zip(q_red.detach().numpy(), q_blue.detach().numpy()):
        strategy_red, strategy_blue = nash_strategies(qr, qb)
        values.append((strategy_red @ qr @ strategy_blue, strategy_red @ qb @ strategy_blue))
    return torch.tensor(values, dtype=torch.float32)


def choose_directions(net, state, explore):
    """One direction index per player: random with probability `explore`,
    otherwise sampled from the Nash strategy of the current Q-values."""
    if random.random() < explore:
        return random.randrange(N_DIRECTIONS), random.randrange(N_DIRECTIONS)
    with torch.no_grad():
        q_red, q_blue = net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    strategy_red, strategy_blue = nash_strategies(q_red[0].numpy(), q_blue[0].numpy())
    pick = lambda p: int(np.random.choice(N_DIRECTIONS, p=np.clip(p, 0, None) / np.clip(p, 0, None).sum()))
    return pick(strategy_red), pick(strategy_blue)


def train_nash_dqn(env, episodes=60, steps=15, batch_size=32, lr=1e-3,
                   target_every=100, explore=(1.0, 0.05), seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net, target = NashQNetwork(), NashQNetwork()
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    buffer, updates = [], 0

    for episode in range(episodes):
        fraction = episode / max(1, episodes - 1)
        epsilon = explore[0] + fraction * (explore[1] - explore[0])
        state = env.reset()
        for _ in range(steps):
            a_red, a_blue = choose_directions(net, state, epsilon)
            next_state, (r_red, r_blue) = env.step(
                (ANGLES[a_red], env.max_step), (ANGLES[a_blue], env.max_step)
            )
            buffer.append((state, a_red, a_blue, r_red, r_blue, next_state))
            state = next_state

            if len(buffer) < batch_size:
                continue
            states, a1, a2, r1, r2, nexts = zip(*random.sample(buffer, batch_size))
            states = torch.as_tensor(np.array(states), dtype=torch.float32)
            nexts = torch.as_tensor(np.array(nexts), dtype=torch.float32)

            with torch.no_grad():
                future = nash_values(*target(nexts))
            rewards = torch.tensor(np.stack([r1, r2], axis=1), dtype=torch.float32)
            goal = rewards + env.discount * future

            q_red, q_blue = net(states)
            index = torch.arange(batch_size)
            taken = torch.stack([q_red[index, a1, a2], q_blue[index, a1, a2]], dim=1)
            loss = nn.functional.mse_loss(taken, goal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            updates += 1
            if updates % target_every == 0:
                target.load_state_dict(net.state_dict())
    return net


def play_greedy(env, net, steps=15):
    """Both players sample from the Nash strategy of the learned Q. Returns final positions."""
    state = env.reset()
    for _ in range(steps):
        a_red, a_blue = choose_directions(net, state, explore=0.0)
        state, _ = env.step((ANGLES[a_red], env.max_step), (ANGLES[a_blue], env.max_step))
    return env.red.copy(), env.blue.copy()
