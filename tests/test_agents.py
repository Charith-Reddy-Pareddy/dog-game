import numpy as np

from doggame.agents import LinearGaussianPolicy, reinforce_update
from doggame.env import Square


def test_policy_mean_stays_within_domain():
    domain = Square(0.0, 1.0)
    policy = LinearGaussianPolicy(domain, seed=0)
    for state in ([0.0, 0.0], [1.0, 1.0], [0.5, 0.2], [-3.0, 9.0]):
        mean = policy.mean(state)
        assert np.all(mean >= domain.low) and np.all(mean <= domain.high)


def test_act_clips_sampled_action_to_domain():
    domain = Square(0.0, 1.0)
    policy = LinearGaussianPolicy(domain, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        action, _, _ = policy.act([0.5, 0.5], rng)
        assert np.all(action >= domain.low) and np.all(action <= domain.high)


def test_zero_advantage_does_not_change_parameters():
    domain = Square(0.0, 1.0)
    policy = LinearGaussianPolicy(domain, seed=0)
    rng = np.random.default_rng(0)
    state = np.array([0.4, 0.6])
    action, mean, std = policy.act(state, rng)

    W_before, b_before, log_std_before = (
        policy.W.copy(),
        policy.b.copy(),
        policy.log_std.copy(),
    )
    reinforce_update(
        policy, [state], [action], [mean], [std], returns=[0.0], lr=0.1, baseline=0.0
    )
    assert np.allclose(policy.W, W_before)
    assert np.allclose(policy.b, b_before)
    assert np.allclose(policy.log_std, log_std_before)


def test_reinforce_update_increases_log_prob_of_rewarded_action():
    domain = Square(0.0, 1.0)
    policy = LinearGaussianPolicy(domain, seed=0)
    rng = np.random.default_rng(1)
    state = np.array([0.3, 0.7])
    action, mean, std = policy.act(state, rng)

    def log_prob():
        m = policy.mean(state)
        s = np.exp(policy.log_std)
        return np.sum(-0.5 * ((action - m) / s) ** 2 - np.log(s))

    before = log_prob()
    reinforce_update(
        policy, [state], [action], [mean], [std], returns=[1.0], lr=0.01, baseline=0.0
    )
    after = log_prob()
    assert after > before
