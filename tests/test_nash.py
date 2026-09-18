import numpy as np

from doggame.env import Square, negative_squared_distance_reward
from doggame.nash import best_response_blue, best_response_red, solve_stage_nash


def test_equilibrium_is_a_mutual_best_response():
    domain = Square(0.0, 1.0)
    action_red, action_blue, _ = solve_stage_nash(
        house_red=(0.9, 0.1), house_blue=(0.1, 0.9), w=0.5, domain=domain
    )
    assert np.allclose(
        best_response_red(action_blue, (0.9, 0.1), 0.5, domain), action_red, atol=1e-6
    )
    assert np.allclose(
        best_response_blue(action_red, (0.1, 0.9), 0.5, domain),
        action_blue,
        atol=1e-6,
    )


def test_symmetric_setup_gives_symmetric_equilibrium():
    domain = Square(0.0, 1.0)
    action_red, action_blue, dog = solve_stage_nash(
        house_red=(0.8, 0.2), house_blue=(0.2, 0.8), w=0.5, domain=domain
    )
    assert np.allclose(dog, dog[::-1], atol=1e-6)


def test_no_unilateral_deviation_improves_payoff():
    domain = Square(0.0, 1.0)
    house_red = np.array([0.9, 0.3])
    house_blue = np.array([0.1, 0.7])
    w = 0.4
    action_red, action_blue, dog = solve_stage_nash(house_red, house_blue, w, domain)
    r_red, r_blue = negative_squared_distance_reward(dog, house_red, house_blue)

    grid = np.linspace(0.0, 1.0, 25)
    candidates = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T

    for candidate in candidates:
        alt_dog = domain.clip(w * candidate + (1 - w) * action_blue)
        alt_r_red, _ = negative_squared_distance_reward(alt_dog, house_red, house_blue)
        assert alt_r_red <= r_red + 1e-9

        alt_dog = domain.clip(w * action_red + (1 - w) * candidate)
        _, alt_r_blue = negative_squared_distance_reward(
            alt_dog, house_red, house_blue
        )
        assert alt_r_blue <= r_blue + 1e-9
