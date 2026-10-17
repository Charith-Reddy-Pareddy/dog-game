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
