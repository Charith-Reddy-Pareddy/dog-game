import numpy as np

from doggame.agents import LinearGaussianPolicy
from doggame.env import DogGameEnv
from doggame.nash import solve_stage_nash
from doggame.train import discounted_returns, train_self_play


def test_discounted_returns_matches_hand_computation():
    returns = discounted_returns([1, 1, 1], discount=0.5)
    assert np.allclose(returns, [1.75, 1.5, 1.0])


def test_training_moves_policy_closer_to_nash():
    env = DogGameEnv(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5, discount=0.9)
    policy_red = LinearGaussianPolicy(env.domain, seed=1)
    policy_blue = LinearGaussianPolicy(env.domain, seed=2)

    state = env.reset()
    nash_red, _, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    initial_distance = np.linalg.norm(policy_red.mean(state) - nash_red)

    train_self_play(
        env, policy_red, policy_blue, episodes=400, steps_per_episode=10, lr=0.2, seed=0
    )

    final_distance = np.linalg.norm(policy_red.mean(state) - nash_red)
    assert final_distance < initial_distance
