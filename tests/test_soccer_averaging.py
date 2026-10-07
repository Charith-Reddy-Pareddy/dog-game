import numpy as np
import torch

from doggame.soccer import Soccer
from doggame.soccer_averaging import SCHEMES, average_from, scheme_exploitabilities
from doggame.soccer_eval import random_policy
from doggame.soccer_solver import solve_soccer

GAME = Soccer(width=4, height=3, goal_size=1)
SOLUTION = solve_soccer(GAME)


def as_snapshots(table, times):
    return [torch.as_tensor(table, dtype=torch.float32) for _ in range(times)]


def test_dropping_weak_early_policies_makes_the_average_less_exploitable():
    # ten rounds of random play, then ten rounds of the exact equilibrium
    history = [as_snapshots(random_policy(GAME), 10) + as_snapshots(SOLUTION.policy0, 10),
               as_snapshots(random_policy(GAME), 10) + as_snapshots(SOLUTION.policy1, 10)]

    scores = scheme_exploitabilities(GAME, history)

    assert scores["second half only"] < 1e-5          # only the equilibrium policies are left
    assert scores["all snapshots"] > 0.2              # the random ones drag the average
    assert scores["all snapshots"] > scores["without the first quarter"] > scores["second half only"] - 1e-9


def test_when_every_snapshot_is_the_same_all_schemes_agree():
    history = [as_snapshots(SOLUTION.policy0, 8), as_snapshots(SOLUTION.policy1, 8)]
    scores = scheme_exploitabilities(GAME, history)
    assert max(scores.values()) < 1e-5


def test_average_from_averages_only_the_snapshots_from_the_start_point():
    a, b, c = (torch.full((3, 4), v) for v in (0.0, 3.0, 6.0))
    assert np.allclose(average_from([a, b, c], 1), 4.5)
    assert np.allclose(average_from([a, b, c], 0), 3.0)


def test_there_is_a_scheme_for_each_part_of_the_history():
    assert list(SCHEMES.values()) == sorted(SCHEMES.values()) and SCHEMES["all snapshots"] == 0.0
