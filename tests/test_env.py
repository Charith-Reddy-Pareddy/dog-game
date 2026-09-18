import numpy as np

from doggame.env import DogGameEnv, Square


def make_env(**kwargs):
    defaults = dict(house_red=(1.0, 1.0), house_blue=(0.0, 0.0), w=0.5)
    defaults.update(kwargs)
    return DogGameEnv(**defaults)


def test_reset_places_dog_between_houses():
    env = make_env(w=0.5)
    dog = env.reset()
    assert np.allclose(dog, [0.5, 0.5])


def test_reset_respects_weight():
    env = make_env(w=0.75)
    dog = env.reset()
    # closer to the red house when w is larger
    assert np.allclose(dog, [0.75, 0.75])


def test_step_moves_dog_to_weighted_average_of_actions():
    env = make_env(w=0.5)
    env.reset()
    next_dog, _, _, _ = env.step(action_red=(1.0, 0.0), action_blue=(0.0, 1.0))
    assert np.allclose(next_dog, [0.5, 0.5])


def test_step_clips_actions_to_domain():
    env = make_env(w=0.5, domain=Square(0.0, 1.0))
    env.reset()
    next_dog, _, _, _ = env.step(action_red=(5.0, 5.0), action_blue=(0.0, 0.0))
    assert np.all(next_dog <= 1.0)
    assert np.allclose(next_dog, [0.5, 0.5])


def test_reward_favors_the_closer_player():
    env = make_env(house_red=(1.0, 1.0), house_blue=(0.0, 0.0), w=0.5)
    env.reset()
    _, (r_red, r_blue), _, _ = env.step(action_red=(1.0, 1.0), action_blue=(1.0, 1.0))
    # dog ends at (1, 1): red should get zero penalty, blue should be penalized
    assert r_red == 0.0
    assert r_blue < 0.0


def test_invalid_weight_raises():
    import pytest

    with pytest.raises(ValueError):
        make_env(w=0.0)
    with pytest.raises(ValueError):
        make_env(w=1.0)
