import numpy as np
import pytest

from doggame.polar import PolarDogGameEnv


def make_env(**kwargs):
    defaults = dict(house_red=(0.8, 0.2), house_blue=(0.2, 0.8), w=0.5, max_step=0.1)
    defaults.update(kwargs)
    return PolarDogGameEnv(**defaults)


def test_players_start_on_their_houses():
    env = make_env()
    state = env.reset()
    assert np.allclose(state[2:4], [0.8, 0.2])
    assert np.allclose(state[4:6], [0.2, 0.8])
    assert np.allclose(state[:2], [0.5, 0.5])


def test_a_move_goes_in_the_angle_direction_by_r():
    env = make_env()
    env.reset()
    env.step((0.0, 0.05), (np.pi, 0.05))  # red goes right, blue goes left
    assert np.allclose(env.red, [0.85, 0.2])
    assert np.allclose(env.blue, [0.15, 0.8])


def test_step_length_is_capped_at_max_step():
    env = make_env(max_step=0.1)
    env.reset()
    env.step((np.pi / 2, 5.0), (0.0, 0.0))  # red asks for 5.0, gets 0.1
    assert np.allclose(env.red, [0.8, 0.3])


def test_players_cannot_leave_the_square():
    env = make_env(house_red=(0.98, 0.5))
    env.reset()
    env.step((0.0, 0.1), (0.0, 0.0))
    assert env.red[0] == 1.0


def test_dog_follows_the_blend_of_positions():
    env = make_env()
    env.reset()
    _, (reward_red, reward_blue) = env.step((0.0, 0.1), (0.0, 0.1))
    assert np.allclose(env.dog, 0.5 * env.red + 0.5 * env.blue)
    assert reward_red == pytest.approx(-np.sum((env.dog - env.house_red) ** 2))


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        make_env(w=1.0)
    with pytest.raises(ValueError):
        make_env(max_step=0.0)
