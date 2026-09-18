"""Self-play training loop.

A single TwoPlayerPolicy (see doggame.agents) is trained against
itself with REINFORCE: rounds are played out under the current
policy, discounted returns are computed per player, and torch.optim
takes the gradient step. We compare the resulting policy against the
analytical stage-game Nash equilibrium from doggame.nash as a sanity
check that the learning dynamics are heading the right way.
"""

import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.env import DogGameEnv
from doggame.nash import solve_stage_nash


def discounted_returns(rewards, discount):
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + discount * running
        returns[t] = running
    return returns


def run_episode(env, policy, steps):
    state = env.reset()
    states, actions_red, actions_blue, rewards_red, rewards_blue = [], [], [], [], []

    for _ in range(steps):
        action_red, action_blue = policy.act(state)
        next_state, (r_red, r_blue), _, _ = env.step(action_red, action_blue)

        states.append(state)
        actions_red.append(action_red)
        actions_blue.append(action_blue)
        rewards_red.append(r_red)
        rewards_blue.append(r_blue)

        state = next_state

    return states, actions_red, actions_blue, rewards_red, rewards_blue


def train_self_play(env, policy, episodes=500, steps_per_episode=20, lr=0.01, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    for _ in range(episodes):
        states, actions_red, actions_blue, rewards_red, rewards_blue = run_episode(
            env, policy, steps_per_episode
        )

        returns_red = discounted_returns(rewards_red, env.discount)
        returns_blue = discounted_returns(rewards_blue, env.discount)
        advantage_red = returns_red - returns_red.mean()
        advantage_blue = returns_blue - returns_blue.mean()

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_red_t = torch.as_tensor(np.array(actions_red), dtype=torch.float32)
        actions_blue_t = torch.as_tensor(np.array(actions_blue), dtype=torch.float32)
        advantage_red_t = torch.as_tensor(advantage_red, dtype=torch.float32)
        advantage_blue_t = torch.as_tensor(advantage_blue, dtype=torch.float32)

        logp_red, logp_blue = policy.log_prob(states_t, actions_red_t, actions_blue_t)
        loss = -(logp_red * advantage_red_t + logp_blue * advantage_blue_t).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append((np.mean(rewards_red), np.mean(rewards_blue)))

    return history


if __name__ == "__main__":
    env = DogGameEnv(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5)
    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=1)

    history = train_self_play(env, policy, episodes=2000, lr=0.01, seed=0)

    nash_red, nash_blue, nash_dog = solve_stage_nash(
        env.house_red, env.house_blue, env.w, env.domain
    )

    test_state = env.reset()
    with torch.no_grad():
        state_t = torch.as_tensor(
            np.asarray(test_state, dtype=np.float32)
        ).unsqueeze(0)
        mean_red, mean_blue = policy.means(state_t)

    print("Learned red action mean: ", mean_red.squeeze(0).numpy())
    print("Nash red action:         ", nash_red)
    print("Learned blue action mean:", mean_blue.squeeze(0).numpy())
    print("Nash blue action:        ", nash_blue)
    print("Average reward, first 10 episodes:", np.mean(history[:10], axis=0))
    print("Average reward, last 10 episodes: ", np.mean(history[-10:], axis=0))
