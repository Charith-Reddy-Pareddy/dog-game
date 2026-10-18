import numpy as np

from doggame.sticky_duopoly import Q_MAX, cournot, deviation_gain, linear_equilibrium, linear_steady_state


def test_without_inertia_the_feedback_equilibrium_is_cournot():
    a, b, _ = linear_equilibrium(0.0)
    assert np.isclose(a, cournot()[0]) and abs(b) < 1e-12


def test_the_linear_policy_is_a_feedback_equilibrium():
    a, b, _ = linear_equilibrium(0.8)
    assert deviation_gain(lambda p: np.clip(a + b * p, 0, Q_MAX), 0.8, grid=401, rounds=200) < 1e-6


def test_sticky_prices_make_the_steady_state_more_competitive_than_cournot():
    outputs = [linear_steady_state(w0)[1] for w0 in (0.3, 0.6, 0.8)]
    assert cournot()[0] < outputs[0] < outputs[1] < outputs[2]
