import numpy as np
import torch

from doggame.continuous_q import (QNetwork, best_response, displacement, equilibrium, play, project,
                                  random_moves, train)
from doggame.dog_game_study import WALK_HOUSES
from doggame.polar import PolarDogGameEnv


class Bowl(torch.nn.Module):
    """A known stage game: each player's Q is minus its squared distance from its own target move."""

    def __init__(self, red=(0.3, -0.4), blue=(-0.5, 0.2), encoding="cartesian"):
        super().__init__()
        self.encoding, self.red, self.blue = encoding, torch.tensor(red), torch.tensor(blue)

    def forward(self, states, p_red, p_blue):
        return -((p_red - self.red) ** 2).sum(-1), -((p_blue - self.blue) ** 2).sum(-1)


def test_projection_keeps_moves_in_each_move_set():
    p = torch.tensor([[2.0, 0.0], [0.3, -0.2], [-3.0, 4.0]])
    assert torch.all(project(p, "cartesian").norm(dim=-1) <= 1 + 1e-6)
    assert torch.all(project(p, "square").abs() <= 1)
    assert torch.allclose(project(p, "cartesian")[1], p[1])
    for encoding in ("cartesian", "polar"):
        steps = np.linalg.norm(displacement(random_moves(200, encoding).numpy(), encoding, 0.2), axis=-1)
        assert steps.max() <= 0.2 + 1e-6


def test_every_best_response_method_finds_the_known_best_move():
    net, state = Bowl(), torch.zeros(6)
    for method in ("grid", "autograd", "finite", "quadratic"):
        move, _, _ = best_response(net, state, torch.zeros(2), 0, method)
        assert torch.allclose(move, net.red, atol=0.06 if method == "grid" else 1e-3), method


def test_simultaneous_gradient_play_reaches_the_known_equilibrium_and_projects_far_targets():
    p_red, p_blue, _ = equilibrium(Bowl(), torch.zeros(1, 6), iters=200)
    assert torch.allclose(p_red[0], torch.tensor([0.3, -0.4]), atol=1e-4)
    far, _, _ = equilibrium(Bowl(red=(3.0, 0.0)), torch.zeros(1, 6), iters=200)
    assert torch.allclose(far[0], torch.tensor([1.0, 0.0]), atol=1e-4)


def test_training_and_greedy_play_run_for_every_encoding():
    env = PolarDogGameEnv(*WALK_HOUSES, max_step=0.2)
    for encoding in ("cartesian", "polar", "square"):
        net = train(env, encoding, episodes=6, steps=15, batch_size=16, seed=0)
        assert isinstance(net, QNetwork) and len(play(env, net, steps=5)) == 6
