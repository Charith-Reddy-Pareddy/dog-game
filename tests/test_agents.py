import numpy as np
import torch

from doggame.agents import TwoPlayerPolicy
from doggame.env import Square


def state_tensor(state):
    return torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)


def test_action_means_stay_within_domain():
    domain = Square(0.0, 1.0)
    policy = TwoPlayerPolicy(domain, architecture="separate", seed=0)
    for state in ([0.0, 0.0], [1.0, 1.0], [0.5, 0.2]):
        mean_red, mean_blue = policy.means(state_tensor(state))
        assert torch.all(mean_red >= domain.low) and torch.all(mean_red <= domain.high)
        assert torch.all(mean_blue >= domain.low) and torch.all(
            mean_blue <= domain.high
        )


def test_act_returns_actions_clipped_to_domain():
    domain = Square(0.0, 1.0)
    policy = TwoPlayerPolicy(domain, architecture="separate", seed=0)
    for _ in range(50):
        action_red, action_blue = policy.act([0.5, 0.5])
        assert np.all(action_red >= domain.low) and np.all(action_red <= domain.high)
        assert np.all(action_blue >= domain.low) and np.all(action_blue <= domain.high)


def test_reinforce_gradient_increases_log_prob_of_rewarded_action():
    domain = Square(0.0, 1.0)
    policy = TwoPlayerPolicy(domain, architecture="separate", seed=0)
    state = [0.3, 0.7]
    action_red, action_blue = policy.act(state)

    st = state_tensor(state)
    action_red_t = torch.as_tensor(action_red, dtype=torch.float32).unsqueeze(0)
    action_blue_t = torch.as_tensor(action_blue, dtype=torch.float32).unsqueeze(0)

    def log_prob_red():
        with torch.no_grad():
            logp_red, _ = policy.log_prob(st, action_red_t, action_blue_t)
        return logp_red.item()

    before = log_prob_red()

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    logp_red, _ = policy.log_prob(st, action_red_t, action_blue_t)
    loss = -logp_red.mean()  # positive advantage: push up log-prob of this action
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = log_prob_red()
    assert after > before


def test_three_architectures_are_all_selectable():
    domain = Square(0.0, 1.0)
    for architecture in ("separate", "shared", "partial"):
        policy = TwoPlayerPolicy(domain, architecture=architecture, seed=0)
        action_red, action_blue = policy.act([0.4, 0.6])
        assert action_red.shape == (2,)
        assert action_blue.shape == (2,)
