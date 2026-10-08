import numpy as np

from doggame.dog_game_trajectories import record
from doggame.polar import PolarDogGameEnv


def test_a_recorded_game_starts_on_the_houses_and_has_a_point_for_every_step():
    env = PolarDogGameEnv((0.9, 0.2), (0.1, 0.8), w=0.5, max_step=0.2)

    path = record(env, lambda state: ((0.0, 0.1), (np.pi, 0.1)), steps=4)

    assert [len(path[name]) for name in ("red", "blue", "dog")] == [5, 5, 5]
    assert path["red"][0] == [0.9, 0.2] and path["blue"][0] == [0.1, 0.8]
    assert path["red"][1] == [1.0, 0.2]  # red walked right by 0.1, up to the edge of the square


def test_players_never_leave_the_square():
    env = PolarDogGameEnv((0.9, 0.2), (0.1, 0.8), w=0.5, max_step=0.2)

    path = record(env, lambda state: ((0.0, 0.2), (np.pi, 0.2)), steps=6)

    for name in ("red", "blue", "dog"):
        assert np.all(np.asarray(path[name]) >= 0.0) and np.all(np.asarray(path[name]) <= 1.0)
