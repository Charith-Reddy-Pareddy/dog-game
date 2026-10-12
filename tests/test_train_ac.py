import numpy as np
import pytest
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.env import DogGameEnv
from doggame.nash import solve_stage_nash
from doggame.train import train_self_play
from doggame.train_ac import ALGORITHMS, train_actor_critic


def make_env():
    return DogGameEnv(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5, discount=0.9)


def red_distance_to_nash(env, policy):
    nash_red, _, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    with torch.no_grad():
        mean_red, _ = policy.means(torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0))
    return np.linalg.norm(mean_red.squeeze(0).numpy() - nash_red)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_training_moves_the_policy_closer_to_nash(algorithm):
    env = make_env()
    policy = TwoPlayerPolicy(env.domain, architecture="separate", seed=1)
    before = red_distance_to_nash(env, policy)

    train_actor_critic(env, policy, algorithm, episodes=400, steps_per_episode=10, lr=0.05, seed=0)

    assert red_distance_to_nash(env, policy) < before


def test_unknown_algorithm_is_rejected():
    env = make_env()
    with pytest.raises(ValueError):
        train_actor_critic(env, TwoPlayerPolicy(env.domain, seed=0), "ddpg", episodes=1)


@pytest.mark.parametrize("train", [
    lambda env, policy, callback: train_self_play(env, policy, episodes=5, steps_per_episode=3, callback=callback),
    lambda env, policy, callback: train_actor_critic(env, policy, "ppo", episodes=5, steps_per_episode=3, callback=callback),
])
def test_callback_is_called_after_every_episode_with_the_count_so_far(train):
    env = make_env()
    seen = []

    train(env, TwoPlayerPolicy(env.domain, seed=0), seen.append)

    assert seen == [1, 2, 3, 4, 5]


def test_same_seed_gives_the_same_policy():
    def trained():
        env = make_env()
        policy = TwoPlayerPolicy(env.domain, seed=3)
        train_actor_critic(env, policy, "a2c", episodes=20, steps_per_episode=5, seed=7)
        return [p.detach().clone() for p in policy.parameters()]

    assert all(torch.equal(a, b) for a, b in zip(trained(), trained()))
