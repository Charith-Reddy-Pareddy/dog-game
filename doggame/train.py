"""Self-play training loop.

Two REINFORCE agents (one per house) learn against each other on the
repeated dog game. We compare the resulting policy against the
analytical stage-game Nash equilibrium from doggame.nash as a sanity
check that the learning dynamics are heading the right way.
"""

import numpy as np

from doggame.agents import LinearGaussianPolicy, reinforce_update
from doggame.env import DogGameEnv
from doggame.nash import solve_stage_nash


def discounted_returns(rewards, discount):
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = (rewards[t] + running) * discount
        returns[t] = running
    return returns


def run_episode(env, policy_red, policy_blue, steps, rng):
    state = env.reset()
    red_log = ([], [], [], [], [])
    blue_log = ([], [], [], [], [])

    for _ in range(steps):
        action_red, mean_red, std_red = policy_red.act(state, rng)
        action_blue, mean_blue, std_blue = policy_blue.act(state, rng)
        next_state, (r_red, r_blue), _, _ = env.step(action_red, action_blue)

        for log, value in zip(
            red_log, (state, action_red, mean_red, std_red, r_red)
        ):
            log.append(value)
        for log, value in zip(
            blue_log, (state, action_blue, mean_blue, std_blue, r_blue)
        ):
            log.append(value)

        state = next_state

    return red_log, blue_log


def train_self_play(
    env,
    policy_red,
    policy_blue,
    episodes=500,
    steps_per_episode=20,
    lr=0.05,
    seed=0,
):
    rng = np.random.default_rng(seed)
    history = []

    for _ in range(episodes):
        red_log, blue_log = run_episode(env, policy_red, policy_blue, steps_per_episode, rng)

        red_states, red_actions, red_means, red_stds, red_rewards = red_log
        blue_states, blue_actions, blue_means, blue_stds, blue_rewards = blue_log

        red_returns = discounted_returns(red_rewards, env.discount)
        blue_returns = discounted_returns(blue_rewards, env.discount)

        reinforce_update(
            policy_red,
            red_states,
            red_actions,
            red_means,
            red_stds,
            red_returns,
            lr,
            baseline=red_returns.mean(),
        )
        reinforce_update(
            policy_blue,
            blue_states,
            blue_actions,
            blue_means,
            blue_stds,
            blue_returns,
            lr,
            baseline=blue_returns.mean(),
        )

        history.append((np.mean(red_rewards), np.mean(blue_rewards)))

    return history


if __name__ == "__main__":
    env = DogGameEnv(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5)
    policy_red = LinearGaussianPolicy(env.domain, seed=1)
    policy_blue = LinearGaussianPolicy(env.domain, seed=2)

    history = train_self_play(env, policy_red, policy_blue, episodes=2000)

    nash_red, nash_blue, nash_dog = solve_stage_nash(
        env.house_red, env.house_blue, env.w, env.domain
    )

    test_state = env.reset()
    print("Learned red action mean: ", policy_red.mean(test_state))
    print("Nash red action:         ", nash_red)
    print("Learned blue action mean:", policy_blue.mean(test_state))
    print("Nash blue action:        ", nash_blue)
    print("Average reward, first 10 episodes:", np.mean(history[:10], axis=0))
    print("Average reward, last 10 episodes: ", np.mean(history[-10:], axis=0))
