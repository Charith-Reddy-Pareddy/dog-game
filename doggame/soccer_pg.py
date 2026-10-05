"""Policy gradient for the soccer game: REINFORCE, A2C and PPO.

Each player is a small network from the board (positions and who has the
ball) to a probability for each of the four moves. Training only ever
plays games (up to `max_steps`, rewards discounted by `discount`); nothing
here solves a game. Two ways to train:

- `train_self_play`: both players learn at once from the same games.
- `train_fictitious_play`: players take turns. The learner trains against a
  mix of the opponent's earlier policies, which is how fictitious play
  reaches an equilibrium: each side best-responds to the other's history.

Policies come out as tables (one row of move probabilities per state), the
form `doggame.soccer_eval` works with.
"""

import numpy as np
import torch
from torch import nn

from doggame.soccer import N_ACTIONS

ALGORITHMS = ("reinforce", "a2c", "ppo")


def _mlp(out, hidden=64):
    return nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class Agent:
    """One player: a policy network, a value network (for A2C and PPO) and an optimizer."""

    def __init__(self, features, algorithm="reinforce", lr=3e-3, entropy=0.001, ppo_clip=0.2, ppo_epochs=4):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        self.features, self.algorithm = features, algorithm
        self.entropy, self.ppo_clip, self.ppo_epochs = entropy, ppo_clip, ppo_epochs
        self.policy, self.value = _mlp(N_ACTIONS), _mlp(1)
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)

    def probs(self, states):
        return torch.softmax(self.policy(self.features[states]), dim=-1)

    def table(self):
        with torch.no_grad():
            return torch.softmax(self.policy(self.features), dim=-1).numpy().astype(np.float64)

    def update(self, states, actions, returns):
        x = self.features[states]
        with torch.no_grad():
            old_logp = torch.log_softmax(self.policy(x), -1).gather(1, actions[:, None]).squeeze(1)
            baseline = self.value(x).squeeze(1) if self.algorithm != "reinforce" else returns.mean()
            advantage = returns - baseline
            if self.algorithm == "ppo":
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.ppo_epochs if self.algorithm == "ppo" else 1):
            log_probs = torch.log_softmax(self.policy(x), -1)
            logp = log_probs.gather(1, actions[:, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
            if self.algorithm == "ppo":
                ratio = (logp - old_logp).exp()
                gain = torch.min(ratio * advantage, ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * advantage).mean()
            else:
                gain = (logp * advantage).mean()
            loss = -gain - self.entropy * entropy
            if self.algorithm != "reinforce":
                loss = loss + 0.5 * nn.functional.mse_loss(self.value(x).squeeze(1), returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def _tensors(game):
    return (torch.as_tensor(game.features(range(game.n_states))),
            torch.as_tensor(game.reward, dtype=torch.float32),
            torch.as_tensor(game.next_state))


def rollout(game, tensors, probs0, probs1, n_games, max_steps, discount):
    """Play `n_games` at once. `probs_i(states)` gives player i's move
    probabilities. Returns the visited states, both players' moves, and
    player 0's discounted return from each step (player 1's is its negative)."""
    _, reward, next_state = tensors
    starts = game.start_states()
    state = torch.tensor([starts[i % 2] for i in range(n_games)])
    active = torch.ones(n_games, dtype=torch.bool)
    visited, moves0, moves1, rewards, live = [], [], [], [], []

    for _ in range(max_steps):
        if not active.any():
            break
        with torch.no_grad():
            a0 = torch.multinomial(probs0(state), 1).squeeze(1)
            a1 = torch.multinomial(probs1(state), 1).squeeze(1)
        nxt = next_state[state, a0, a1]
        visited.append(state)
        moves0.append(a0)
        moves1.append(a1)
        rewards.append(reward[state, a0, a1])
        live.append(active.clone())
        active = active & (nxt != game.n_states)
        state = torch.where(nxt == game.n_states, state, nxt)

    running = torch.zeros(n_games)
    returns = [None] * len(visited)
    for t in reversed(range(len(visited))):
        running = live[t] * (rewards[t] + discount * running)
        returns[t] = running
    keep = torch.stack(live).flatten()
    flat = lambda xs: torch.stack(xs).flatten()[keep]
    return flat(visited), flat(moves0), flat(moves1), flat(returns)


def train_self_play(game, algorithm="reinforce", iterations=300, games=256, max_steps=100,
                    discount=0.9, seed=0, on_iteration=None, **agent_options):
    """Both players learn at once from the same games. Returns (agent0, agent1).
    `on_iteration(i, agents)`, if given, is called after each update."""
    torch.manual_seed(seed)
    tensors = _tensors(game)
    agents = [Agent(tensors[0], algorithm, **agent_options) for _ in range(2)]
    for i in range(iterations):
        states, a0, a1, returns = rollout(game, tensors, agents[0].probs, agents[1].probs, games, max_steps, discount)
        agents[0].update(states, a0, returns)
        agents[1].update(states, a1, -returns)
        if on_iteration:
            on_iteration(i, agents)
    return agents


def train_fictitious_play(game, algorithm="reinforce", rounds=20, iterations=15, games=256, max_steps=100,
                          discount=0.9, seed=0, on_round=None, **agent_options):
    """Players take turns learning a best response (by policy gradient) to a
    uniform mix of the opponent's earlier policies. Returns (average policy of
    player 0, average policy of player 1), each a table over states.
    `on_round(i, history)`, if given, is called after each round with every
    snapshot so far."""
    torch.manual_seed(seed)
    tensors = _tensors(game)
    agents = [Agent(tensors[0], algorithm, **agent_options) for _ in range(2)]
    history = [[torch.as_tensor(a.table(), dtype=torch.float32)] for a in agents]  # snapshots of each player

    def mixture(player):
        """Move probabilities from a different earlier snapshot in each game."""
        snapshots = torch.stack(history[player])
        picks = torch.randint(len(snapshots), (games,))
        return lambda states: snapshots[picks, states]

    for r in range(rounds):
        for learner in (0, 1):
            for _ in range(iterations):
                opponent = mixture(1 - learner)
                if learner == 0:
                    states, a0, _, returns = rollout(game, tensors, agents[0].probs, opponent, games, max_steps, discount)
                    agents[0].update(states, a0, returns)
                else:
                    states, _, a1, returns = rollout(game, tensors, opponent, agents[1].probs, games, max_steps, discount)
                    agents[1].update(states, a1, -returns)
            history[learner].append(torch.as_tensor(agents[learner].table(), dtype=torch.float32))
        if on_round:
            on_round(r, history)
    return tuple(torch.stack(h).mean(0).numpy().astype(np.float64) for h in history)
