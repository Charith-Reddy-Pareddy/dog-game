import numpy as np

from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_dp import solve_1d
from doggame.inertia_mpe import one_shot_gain
from doggame.inertia_theory import deviation_gain, explicit_policy, hold_share, holds_everywhere, nash_1d


def numeric_share(h_red, h_blue, w, a_red, a_blue, w0, g, n=201):
    """Share of dog positions where a player gains by one round of deviation from holding (1D slice of the 2D check)."""
    bad = 0
    for x in np.linspace(0, 1, n):
        gain_red = one_shot_gain([x], [h_red], [a_red], [a_blue], w, w0, g)[0]
        gain_blue = one_shot_gain([x], [h_blue], [a_blue], [a_red], 1 - w, w0, g)[0]
        bad += max(gain_red, gain_blue) > 1e-12
    return bad / n


def test_the_closed_form_hold_share_matches_the_numerical_one_round_check():
    rng = np.random.default_rng(0)
    for _ in range(40):
        h_red, h_blue, w, w0, g = *rng.random(2), rng.uniform(0.1, 0.9), rng.uniform(0, 0.95), rng.uniform(0.1, 0.99)
        a_red, a_blue, _ = nash_1d(h_red, h_blue, w)
        share = hold_share(([h_red], [h_blue]), w, w0, ([a_red], [a_blue]), g)[0]
        assert abs(share - numeric_share(h_red, h_blue, w, a_red, a_blue, w0, g)) < 0.011


def test_holding_is_markov_perfect_in_setup_1_up_to_moderate_inertia_only():
    c = DEFAULT_CONFIGS[0]
    houses, actions = (c["house_red"], c["house_blue"]), ((1, 0), (0, 1))
    assert holds_everywhere(houses, c["w"], 0.6, actions) and not holds_everywhere(houses, c["w"], 0.9, actions)


def test_the_explicit_policy_holds_the_one_shot_actions_near_the_one_shot_dog():
    act = explicit_policy(0.8, 0.2, 0.5, 1.0, 0.0, 0.9)  # setup 1's y coordinate, mirrored
    assert np.allclose(act(0.5), (1.0, 0.0)) and act(1.0)[0] < 1.0 and act(0.0)[1] > 0.0


def test_no_one_round_deviation_from_the_explicit_policy_gains_when_the_houses_straddle_the_dog():
    rng = np.random.default_rng(1)
    for _ in range(8):
        h_red, h_blue = sorted(rng.random(2))[::-1]
        w, w0, g = rng.uniform(h_blue, h_red), rng.uniform(0.1, 0.95), rng.uniform(0.3, 0.99)
        a_red, a_blue, kind = nash_1d(h_red, h_blue, w)
        act = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0, g)
        assert kind == "straddle"
        for x in np.linspace(0, 1, 9):
            assert deviation_gain(act, h_red, w, w0, x, True, g, grid=401) < 1e-12
            assert deviation_gain(act, h_blue, w, w0, x, False, g, grid=401) < 1e-12


def test_the_explicit_policy_can_fail_when_a_one_shot_action_is_interior():
    h_red, h_blue, w, w0, g = 0.754, 0.538, 0.347, 0.767, 0.335  # blue's edge would overshoot its house
    a_red, a_blue, kind = nash_1d(h_red, h_blue, w)
    act = explicit_policy(h_red, h_blue, w, a_red, a_blue, w0, g)
    assert kind == "interior"
    assert max(deviation_gain(act, h, w, w0, x, red, g, grid=401) for x in np.linspace(0, 1, 41) for h, red in ((h_red, True), (h_blue, False))) > 1e-3


def test_the_explicit_policy_matches_the_dynamic_program():
    c, w0 = DEFAULT_CONFIGS[2], 0.6
    h_red, h_blue = c["house_red"][0], c["house_blue"][0]
    sol = solve_1d(h_red, h_blue, c["w"], w0, states=81, actions=51)
    red, blue = explicit_policy(h_red, h_blue, c["w"], *nash_1d(h_red, h_blue, c["w"])[:2], w0)(sol["z"])
    assert max(np.abs(red - sol["red"]).max(), np.abs(blue - sol["blue"]).max()) <= 0.03
