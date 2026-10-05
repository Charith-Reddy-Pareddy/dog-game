import numpy as np

from doggame.soccer import MIRROR_ACTION, Soccer
from doggame.soccer_eval import best_response, exploitability, play_games, random_policy, symmetry_gap
from doggame.soccer_solver import solve_soccer

GAME = Soccer(width=5, height=3, goal_size=1)
SOLUTION = solve_soccer(GAME)


def test_the_exact_solution_has_no_exploitability():
    assert abs(exploitability(GAME, SOLUTION.policy0, SOLUTION.policy1)) < 1e-6


def test_random_play_is_exploitable():
    uniform = random_policy(GAME)
    assert exploitability(GAME, uniform, uniform) > 0.5


def test_the_best_response_to_random_play_beats_random_play():
    uniform = random_policy(GAME)
    best, _ = best_response(GAME, uniform, player=0)
    result = play_games(GAME, best, uniform, n_games=400)
    assert result["wins"] > 3 * result["losses"]


def test_best_response_to_the_exact_solution_cannot_win():
    best, _ = best_response(GAME, SOLUTION.policy1, player=0)
    result = play_games(GAME, best, SOLUTION.policy1, n_games=200)
    assert result["wins"] == 0


def test_counts_add_up_and_equilibrium_play_ties():
    result = play_games(GAME, SOLUTION.policy0, SOLUTION.policy1, n_games=100)
    assert sum(result.values()) == 100
    assert result["ties"] == 100


def test_a_policy_and_its_own_mirror_image_have_no_symmetry_gap():
    rng = np.random.default_rng(0)
    policy0 = rng.dirichlet(np.ones(4), size=GAME.n_states)
    policy1 = np.empty_like(policy0)
    policy1[GAME.mirror()] = policy0[:, MIRROR_ACTION]

    assert symmetry_gap(GAME, policy0, policy1) < 1e-12


def test_two_lopsided_policies_have_a_symmetry_gap():
    always_north = np.tile([1.0, 0, 0, 0], (GAME.n_states, 1))
    always_south = np.tile([0, 1.0, 0, 0], (GAME.n_states, 1))
    assert symmetry_gap(GAME, always_north, always_south) > 1.0
