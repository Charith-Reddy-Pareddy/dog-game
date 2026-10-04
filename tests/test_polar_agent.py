import numpy as np
import torch

from doggame.env import Square
from doggame.nash import solve_stage_nash
from doggame.polar import PolarDogGameEnv
from doggame.polar_agent import PolarPolicy, final_positions, train_polar


def test_act_returns_an_angle_and_a_unit_step_for_each_player():
    policy = PolarPolicy(seed=0)
    for _ in range(20):
        move_red, move_blue = policy.act(np.full(6, 0.5))
        for theta, r in (move_red, move_blue):
            assert -np.pi <= theta <= np.pi
            assert 0.0 <= r <= 1.0


def test_an_angle_and_the_same_angle_plus_a_full_turn_are_equally_likely():
    policy = PolarPolicy(seed=0)
    states = torch.full((1, 6), 0.5)
    moves = torch.tensor([[0.7, 0.4]])
    turned = torch.tensor([[0.7 + 2 * np.pi, 0.4]])

    same, _ = policy.log_prob(states, moves, moves)
    wrapped, _ = policy.log_prob(states, turned, turned)

    assert torch.allclose(same, wrapped, atol=1e-4)


def test_training_walks_the_players_to_the_nash_corners():
    env = PolarDogGameEnv((0.88, 0.12), (0.3, 0.65), w=0.5, max_step=0.2)
    nash_red, nash_blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, Square())
    policy = PolarPolicy(seed=0)

    def average_distance():
        total = 0.0
        for _ in range(10):
            red, blue = final_positions(env, policy)
            total += np.linalg.norm(red - nash_red) + np.linalg.norm(blue - nash_blue)
        return total / 10

    before = average_distance()
    train_polar(env, policy, episodes=300)

    assert average_distance() < 0.25 * before
