import numpy as np

from doggame.env import DogGameEnv, Square, make_inertial_transition, weighted_average_transition


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


def test_inertial_transition_with_zero_inertia_matches_default():
    transition = make_inertial_transition(0.0)
    dog = np.array([0.2, 0.9])
    action_red = np.array([1.0, 0.0])
    action_blue = np.array([0.0, 1.0])
    for w in (0.1, 0.5, 0.8):
        assert np.allclose(
            transition(dog, action_red, action_blue, w),
            weighted_average_transition(dog, action_red, action_blue, w),
        )


def test_inertial_transition_only_partially_moves_each_round():
    env = make_env(w=0.5, transition_fn=make_inertial_transition(0.6))
    env.reset()  # dog starts at (0.5, 0.5)
    next_dog, _, _, _ = env.step(action_red=(1.0, 0.0), action_blue=(0.0, 1.0))
    # the fully re-targeted position would be (0.5, 0.5) (unchanged here,
    # since the target happens to coincide with the start); use a
    # target away from the start to see partial movement instead
    env2 = make_env(house_red=(0.9, 0.9), house_blue=(0.9, 0.9), w=0.5,
                     transition_fn=make_inertial_transition(0.6))
    start = env2.reset()  # dog starts at (0.9, 0.9)
    moved, _, _, _ = env2.step(action_red=(0.0, 0.0), action_blue=(0.0, 0.0))
    # target is (0, 0); with w0 = 0.6 the dog should move only 40% of the way
    expected = 0.6 * start + 0.4 * np.array([0.0, 0.0])
    assert np.allclose(moved, expected)
    assert not np.allclose(moved, [0.0, 0.0])


def test_inertial_transition_converges_to_the_same_fixed_point_as_no_inertia():
    # holding actions fixed, inertia should only slow convergence, not
    # change the eventual dog position
    house_red, house_blue, w = (0.9, 0.2), (0.1, 0.8), 0.5
    action_red, action_blue = (1.0, 0.0), (0.0, 1.0)

    env_fast = make_env(house_red=house_red, house_blue=house_blue, w=w)
    env_slow = make_env(house_red=house_red, house_blue=house_blue, w=w,
                         transition_fn=make_inertial_transition(0.7))
    env_fast.reset()
    env_slow.reset()

    for _ in range(200):
        fast_dog, _, _, _ = env_fast.step(action_red, action_blue)
        slow_dog, _, _, _ = env_slow.step(action_red, action_blue)

    assert np.allclose(fast_dog, slow_dog, atol=1e-6)


def test_inertial_transition_rejects_out_of_range_w0():
    import pytest

    with pytest.raises(ValueError):
        make_inertial_transition(-0.1)
    with pytest.raises(ValueError):
        make_inertial_transition(1.0)
