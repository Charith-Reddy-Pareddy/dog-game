"""Multi-agent policy network architectures.

The meeting discussed three ways to structure a two-player policy
network:

1. "separate" -- two fully independent networks, one per player.
2. "shared"   -- a single shared trunk that both players' heads read
                 from, with separate output heads ("everything is
                 shared... we still need different weights to output").
3. "partial"  -- a shared trunk feeding into separate per-player
                 branches. This is structurally equivalent to one
                 joint network with the cross-player weights frozen
                 to zero, but simpler to express directly as two
                 branches off a shared backbone.

All three expose the same forward(state) -> (mean_red, mean_blue)
API so training code doesn't need to know which one it's using.
"""

from torch import nn


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.out(self.body(x))


class SeparatePolicyNetwork(nn.Module):
    """Two fully independent networks, one per player."""

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=16):
        super().__init__()
        self.red = _MLP(state_dim, hidden_dim, action_dim)
        self.blue = _MLP(state_dim, hidden_dim, action_dim)

    def forward(self, state):
        return self.red(state), self.blue(state)


class SharedTrunkPolicyNetwork(nn.Module):
    """A single shared trunk feeding two separate output heads."""

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=16):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.head_red = nn.Linear(hidden_dim, action_dim)
        self.head_blue = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.trunk(state)
        return self.head_red(h), self.head_blue(h)


class PartiallySharedPolicyNetwork(nn.Module):
    """A shared trunk followed by separate per-player branches."""

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=16, branch_hidden=8):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.branch_red = _MLP(hidden_dim, branch_hidden, action_dim)
        self.branch_blue = _MLP(hidden_dim, branch_hidden, action_dim)

    def forward(self, state):
        h = self.trunk(state)
        return self.branch_red(h), self.branch_blue(h)


ARCHITECTURES = {
    "separate": SeparatePolicyNetwork,
    "shared": SharedTrunkPolicyNetwork,
    "partial": PartiallySharedPolicyNetwork,
}


def make_policy_network(architecture, **kwargs):
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}, choose from {list(ARCHITECTURES)}"
        )
    return ARCHITECTURES[architecture](**kwargs)
