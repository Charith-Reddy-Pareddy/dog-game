import itertools

import numpy as np

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS
from doggame.nash import solve_stage_nash
from doggame.potential import potential, rewards, slope_along_flat_move

RNG = np.random.default_rng(0)


def test_a_one_player_deviation_changes_the_potential_by_exactly_that_players_reward_change():
    for config in DEFAULT_CONFIGS:
        h_red, h_blue, w = config["house_red"], config["house_blue"], config["w"]
        for _ in range(50):
            a_red, a_blue, other = RNG.random(2), RNG.random(2), RNG.random(2)
            phi = potential(a_red, a_blue, h_red, h_blue, w)
            red_gain = rewards(other, a_blue, h_red, h_blue, w)[0] - rewards(a_red, a_blue, h_red, h_blue, w)[0]
            blue_gain = rewards(a_red, other, h_red, h_blue, w)[1] - rewards(a_red, a_blue, h_red, h_blue, w)[1]
            assert np.isclose(red_gain, potential(other, a_blue, h_red, h_blue, w) - phi)
            assert np.isclose(blue_gain, potential(a_red, other, h_red, h_blue, w) - phi)


def test_the_analytical_equilibrium_maximizes_the_potential_when_it_is_unique():
    axis = np.linspace(0, 1, 11)
    grid = list(itertools.product(axis, repeat=2))
    for index in (0, 1, 2, 4):  # setup 3 has a continuum of equilibria
        c = DEFAULT_CONFIGS[index]
        nash_red, nash_blue, _ = solve_stage_nash(c["house_red"], c["house_blue"], c["w"], Square(0.0, 1.0))
        best = max(potential(r, b, c["house_red"], c["house_blue"], c["w"]) for r in grid for b in grid)
        assert np.isclose(potential(nash_red, nash_blue, c["house_red"], c["house_blue"], c["w"]), best)


def test_the_potential_is_flat_exactly_where_the_houses_share_a_coordinate():
    shared = DEFAULT_CONFIGS[3]  # both houses at y = 0.5
    slope = slope_along_flat_move(shared["house_red"], shared["house_blue"], shared["w"])

    assert slope[0] > 0 and slope[1] == 0


def test_the_potential_cannot_tell_the_interior_equilibrium_from_the_two_boundary_ones():
    c = DEFAULT_CONFIGS[3]
    phi = lambda red, blue: potential(red, blue, c["house_red"], c["house_blue"], c["w"])

    interior = phi((1, 0.5), (0, 0.5))      # what iterated best response returns
    red_top = phi((1, 1.0), (0, 0.0))       # where most self-play runs end
    red_bottom = phi((1, 0.0), (0, 1.0))    # the mirror image
    assert np.isclose(interior, red_top) and np.isclose(interior, red_bottom)


def test_the_smoothed_potential_tracks_expected_reward_changes_and_dips_at_the_interior():
    from doggame.potential import clipped_moments, smoothed_potential

    rng = np.random.default_rng(0)
    draws = np.clip(rng.normal(0.8, 0.4, 1_000_000), 0, 1)
    assert np.allclose(clipped_moments(0.8, 0.4), (draws.mean(), (draws ** 2).mean()), atol=2e-3)

    def expected_reward(mean_red, mean_blue, h=0.5, w=0.5, std=0.4, n=400_000):
        a, b = (np.clip(rng.normal(m, std, n), 0, 1) for m in (mean_red, mean_blue))
        return -np.mean((w * a + (1 - w) * b - h) ** 2)

    change = smoothed_potential(0.9, 0.1, 0.5, 0.5, 0.5, 0.4) - smoothed_potential(0.3, 0.1, 0.5, 0.5, 0.5, 0.4)
    assert abs(change - (expected_reward(0.9, 0.1) - expected_reward(0.3, 0.1))) < 2e-3
    line = [smoothed_potential(t, 1 - t, 0.5, 0.5, 0.5, 0.37) for t in np.linspace(0, 1, 11)]
    assert np.argmin(line) == 5 and np.isclose(line[0], line[-1])
