import random

import numpy as np
import torch

from doggame.env import Square
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.polar_dqn import N_DIRECTIONS, NashQNetwork, choose_directions, nash_values, play_greedy, train_nash_dqn


def test_network_gives_a_q_matrix_per_player():
    q_red, q_blue = NashQNetwork()(torch.zeros(3, 6))
    assert q_red.shape == (3, N_DIRECTIONS, N_DIRECTIONS)
    assert q_blue.shape == (3, N_DIRECTIONS, N_DIRECTIONS)


def test_nash_value_of_a_symmetric_zero_sum_state_is_zero():
    # an antisymmetric payoff matrix is a fair game: both players' value is 0
    m = torch.randn(1, N_DIRECTIONS, N_DIRECTIONS, generator=torch.Generator().manual_seed(0))
    antisymmetric = m - m.transpose(1, 2)

    values = nash_values(antisymmetric, -antisymmetric)

    assert values.shape == (1, 2)
    assert torch.allclose(values, torch.zeros(1, 2), atol=1e-5)


def test_directions_are_valid_indices():
    net = NashQNetwork()
    for explore in (0.0, 1.0):
        for _ in range(5):
            a_red, a_blue = choose_directions(net, np.full(6, 0.5), explore)
            assert 0 <= a_red < N_DIRECTIONS and 0 <= a_blue < N_DIRECTIONS


def test_training_moves_players_toward_the_nash_corners():
    env = PolarDogGameEnv((0.88, 0.12), (0.3, 0.65), w=0.5, max_step=0.2)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, Square())

    def average_distance(net):
        total = 0.0
        for _ in range(6):
            red, blue = play_greedy(env, net)
            total += np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)
        return total / 6

    random.seed(0)  # the evaluation games sample moves, so seed them too
    np.random.seed(0)
    torch.manual_seed(0)
    before = average_distance(NashQNetwork())
    after = average_distance(train_nash_dqn(env, episodes=25, seed=0))

    assert after < 0.5 * before


def test_training_twice_with_the_same_seed_gives_the_same_network():
    # a time limit on the equilibrium solver once made this vary with machine load
    env = PolarDogGameEnv((0.88, 0.12), (0.3, 0.65), w=0.5, max_step=0.2)
    first = train_nash_dqn(env, episodes=4, seed=3)
    second = train_nash_dqn(env, episodes=4, seed=3)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
