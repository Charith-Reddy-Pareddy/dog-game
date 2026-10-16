import numpy as np
import torch

from doggame.agents import BoundedCartesianPolicy, PolarActionPolicy
from doggame.env import DogGameEnv, Square
from doggame.train import train_self_play
from doggame.train_ac import train_actor_critic


def test_polar_mapping_round_trips_and_stays_inside_the_square():
    policy = PolarActionPolicy(Square(0.0, 1.0), seed=1)
    theta = torch.linspace(-np.pi, np.pi, 101)
    fraction = torch.linspace(0.0, 1.0, 101)
    actions = policy._to_action(theta, fraction)
    recovered_theta, recovered_fraction = policy._from_action(actions)
    angle_error = torch.atan2(torch.sin(recovered_theta - theta), torch.cos(recovered_theta - theta))
    nonzero = fraction > 0
    assert torch.allclose(recovered_fraction, fraction, atol=1e-6)
    assert torch.allclose(angle_error[nonzero], torch.zeros_like(angle_error[nonzero]), atol=3e-6)
    assert torch.all(actions >= 0.0) and torch.all(actions <= 1.0)


def test_polar_policy_samples_feasible_points_and_has_finite_gradients():
    policy = PolarActionPolicy(Square(0.0, 1.0), seed=2)
    state = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    red, blue = policy.sample_action_batch(state, samples=128, seed=5)
    assert red.shape == blue.shape == (128, 2, 2)
    assert torch.all(red >= 0.0) and torch.all(red <= 1.0)
    assert torch.all(blue >= 0.0) and torch.all(blue <= 1.0)

    red_logp, blue_logp = policy.log_prob(state, red[0], blue[0])
    loss = -(red_logp + blue_logp).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())


def test_polar_policy_is_trainable_by_reinforce_a2c_and_ppo():
    env = DogGameEnv((0.9, 0.2), (0.1, 0.8), w=0.5, discount=0.9)
    for algorithm in ("reinforce", "a2c", "ppo"):
        policy = PolarActionPolicy(env.domain, seed=7)
        before = [p.detach().clone() for p in policy.parameters()]
        if algorithm == "reinforce":
            train_self_play(env, policy, episodes=2, steps_per_episode=3, lr=0.01, seed=3)
        else:
            train_actor_critic(
                env, policy, algorithm, episodes=2, steps_per_episode=3, lr=0.01,
                seed=3, ppo_epochs=2,
            )
        assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))


def test_polar_expected_actions_are_reproducible_and_feasible():
    policy = PolarActionPolicy(Square(-2.0, 4.0), seed=3)
    states = torch.tensor([[0.1, 0.2], [0.6, 0.8]])
    red_1, blue_1 = policy.means(states)
    red_2, blue_2 = policy.means(states)
    assert torch.equal(red_1, red_2) and torch.equal(blue_1, blue_2)
    for actions in (red_1, blue_1):
        assert torch.all(actions >= -2.0) and torch.all(actions <= 4.0)


def test_bounded_cartesian_beta_policy_is_supported_by_all_pg_losses():
    env = DogGameEnv((0.9, 0.2), (0.1, 0.8), w=0.5, discount=0.9)
    policy = BoundedCartesianPolicy(env.domain, seed=8)
    red, blue = policy.sample_action_batch(torch.tensor([[0.5, 0.5]]), samples=32, seed=9)
    assert red.shape == blue.shape == (32, 1, 2)
    assert torch.all(red >= 0.0) and torch.all(red <= 1.0)
    logp_red, logp_blue = policy.log_prob(torch.tensor([[0.5, 0.5]]), red[0], blue[0])
    assert torch.isfinite(logp_red).all() and torch.isfinite(logp_blue).all()
    train_self_play(env, policy, episodes=1, steps_per_episode=2, lr=0.01, seed=4)
