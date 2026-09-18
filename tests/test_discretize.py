import numpy as np

from doggame.discretize import action_grid, payoff_matrices
from doggame.env import Square, negative_squared_distance_reward, weighted_average_transition


def test_action_grid_covers_domain_bounds():
    domain = Square(0.0, 1.0)
    grid = action_grid(domain, resolution=3)
    assert grid.shape == (9, 2)
    assert np.isclose(grid.min(), 0.0)
    assert np.isclose(grid.max(), 1.0)


def test_payoff_matrices_match_hand_computed_values():
    domain = Square(0.0, 1.0)
    grid = action_grid(domain, resolution=2)  # corners only: (0,0),(0,1),(1,0),(1,1)
    house_red = np.array([1.0, 1.0])
    house_blue = np.array([0.0, 0.0])
    w = 0.5

    payoff_red, payoff_blue = payoff_matrices(
        grid,
        house_red,
        house_blue,
        w,
        domain,
        weighted_average_transition,
        negative_squared_distance_reward,
    )

    # find the row/col for the two opposite corners
    i_00 = np.where((grid == [0.0, 0.0]).all(axis=1))[0][0]
    i_11 = np.where((grid == [1.0, 1.0]).all(axis=1))[0][0]

    # red plays its own house, blue plays its own house -> dog at the midpoint
    dog = 0.5 * house_red + 0.5 * house_blue
    expected_r_red, expected_r_blue = negative_squared_distance_reward(
        dog, house_red, house_blue
    )
    assert np.isclose(payoff_red[i_11, i_00], expected_r_red)
    assert np.isclose(payoff_blue[i_11, i_00], expected_r_blue)

    # both players play red's house -> dog lands exactly on red's house
    assert np.isclose(payoff_red[i_11, i_11], 0.0)
    assert payoff_blue[i_11, i_11] < 0.0
