import numpy as np

from doggame.dog_game_study import WALK_HOUSES
from doggame.env import Square
from doggame.nash import solve_stage_nash
from doggame.walking_equilibrium import check_profile, corner_path, discounted_reward

CORNERS = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))


def test_the_corner_path_walks_at_full_speed_reaches_the_corner_and_stays():
    path = corner_path((0.3, 0.65), (0.0, 1.0), 0.2, 6)

    moves = np.linalg.norm(np.diff(np.vstack([(0.3, 0.65), path]), axis=0), axis=1)
    assert np.allclose(moves[:2], 0.2) and moves.max() <= 0.2 + 1e-12
    assert np.allclose(path[2:], (0.0, 1.0))  # 0.46 away: two full steps, a short third, then it stays


def test_the_reward_is_minus_the_discounted_squared_distance_from_the_dog_to_the_house():
    path, other = np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[1.0, 1.0], [1.0, 1.0]])

    # the dog is at (0.5, 0.5) each step, 0.5 away from the house at the origin: -(0.5 + 0.9 * 0.5)
    assert np.isclose(discounted_reward(path, other, np.array([0.0, 0.0]), 0.5, 0.5, 0.9), -(0.5 + 0.9 * 0.5))


def test_both_players_resting_on_the_one_shot_corners_is_an_exact_equilibrium():
    houses = WALK_HOUSES
    corners = tuple(solve_stage_nash(houses[0], houses[1], 0.5, Square())[:2])

    result = check_profile(houses, 0.5, corners, starts=corners)

    assert result["red"]["gain"] < 1e-9 and result["blue"]["gain"] < 1e-9


def test_walking_straight_to_the_corner_is_not_exactly_a_best_response_but_the_gain_is_small():
    result = check_profile(WALK_HOUSES, 0.5, CORNERS)

    assert result["red"]["gain"] < 1e-6                            # red's best response is the straight walk
    assert 1e-4 < result["blue"]["gain"] < 0.01                    # blue gains a little by curving toward the unclipped optimum
    assert result["blue"]["best"] >= result["blue"]["followed"]
