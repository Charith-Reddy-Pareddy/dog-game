import numpy as np

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_equilibrium import check_profile, discounted_reward
from doggame.nash import solve_stage_nash

CONFIG = DEFAULT_CONFIGS[0]
HOUSES = (CONFIG["house_red"], CONFIG["house_blue"])


def nash():
    return tuple(solve_stage_nash(HOUSES[0], HOUSES[1], CONFIG["w"], Square(0.0, 1.0))[:2])


def test_the_reward_follows_the_dog_which_keeps_part_of_its_old_position():
    # one round, dog starts at (1, 1), w0 = 0.5: dog' = 0.5 * (1, 1) + 0.5 * (0.5 * (0, 0) + 0.5 * (0, 0)) = (0.5, 0.5); house at the origin
    reward = discounted_reward(np.array([[0.0, 0.0]]), np.array([0.0, 0.0]), 0.5, 0.5, np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.9)

    assert np.isclose(reward, -0.5)


def test_the_one_shot_nash_profile_has_no_profitable_deviation_with_or_without_inertia():
    for w0 in (0.0, 0.6):
        result = check_profile(HOUSES, CONFIG["w"], w0, nash())

        assert result["red"]["gain"] < 1e-9 and result["blue"]["gain"] < 1e-9


def test_the_check_does_find_a_gain_when_the_profile_is_not_an_equilibrium():
    # This opponent action changes the best response; the previous midpoint
    # control accidentally left red's action as a best response in this setup.
    result = check_profile(HOUSES, CONFIG["w"], 0.6, (nash()[0], np.array([0.0, 0.0])))

    assert result["red"]["gain"] > 1e-3
