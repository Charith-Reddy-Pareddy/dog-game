import numpy as np

from doggame.env import make_inertial_transition
from doggame.experiments import DEFAULT_CONFIGS
from doggame.feedback_learning import RandomStart, explicit_2d, feedback_gain, holding_2d


def test_episodes_start_from_random_dog_positions():
    env = RandomStart(**DEFAULT_CONFIGS[0], transition_fn=make_inertial_transition(0.6), seed=0)
    starts = np.array([env.reset() for _ in range(20)])
    assert starts.min() >= 0 and starts.max() <= 1 and len(np.unique(starts[:, 0])) == 20


def test_the_explicit_equilibrium_has_no_feedback_gain_and_holding_does():
    c = DEFAULT_CONFIGS[2]
    small = dict(states=5, grid=11, rounds=60)
    assert feedback_gain(explicit_2d(c, 0.6), c, 0.6, **small)[0] < 1e-12
    assert feedback_gain(holding_2d(c), c, 0.6, **small)[0] > 0.01
