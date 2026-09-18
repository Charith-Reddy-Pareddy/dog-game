import pytest
import torch

from doggame.networks import ARCHITECTURES, make_policy_network


def test_all_architectures_produce_correct_output_shapes():
    state = torch.zeros(4, 2)
    for name in ARCHITECTURES:
        net = make_policy_network(name)
        mean_red, mean_blue = net(state)
        assert mean_red.shape == (4, 2)
        assert mean_blue.shape == (4, 2)


def test_separate_architecture_has_no_shared_parameters():
    net = make_policy_network("separate")
    red_params = {id(p) for p in net.red.parameters()}
    blue_params = {id(p) for p in net.blue.parameters()}
    assert red_params.isdisjoint(blue_params)


def test_shared_architecture_uses_identical_trunk_for_both_heads():
    net = make_policy_network("shared")
    state = torch.randn(1, 2)
    mean_red, _ = net(state)
    mean_red.sum().backward()
    # the trunk sits on the path from state to mean_red, so it must
    # receive a gradient even though only the red head was used
    assert all(p.grad is not None for p in net.trunk.parameters())
    assert all(p.grad is None for p in net.head_blue.parameters())


def test_partial_architecture_shares_trunk_but_not_branches():
    net = make_policy_network("partial")
    state = torch.randn(1, 2)
    mean_red, _ = net(state)
    mean_red.sum().backward()
    assert all(p.grad is not None for p in net.trunk.parameters())
    assert all(p.grad is not None for p in net.branch_red.parameters())
    assert all(p.grad is None for p in net.branch_blue.parameters())


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        make_policy_network("nonsense")
