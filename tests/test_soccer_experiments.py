from doggame.soccer import Soccer
from doggame.soccer_eval import random_policy
from doggame.soccer_experiments import counts, evaluate
from doggame.soccer_solver import solve_soccer

GAME = Soccer(width=5, height=3, goal_size=1)
SOLUTION = solve_soccer(GAME)


def test_the_exact_solution_scores_as_unexploitable_and_never_loses():
    assert SOLUTION.mixed.sum() == 0  # pure equilibrium, so every game against it is decided by its value
    result = evaluate(GAME, SOLUTION, SOLUTION.policy0, SOLUTION.policy1, n_games=200)
    assert abs(result["exploitability"]) < 1e-6
    for opponent in ("random", "nash", "best response"):
        assert sum(result[opponent].values()) == 200 and result[opponent]["losses"] == 0


def test_random_play_is_scored_as_exploitable():
    uniform = random_policy(GAME)
    result = evaluate(GAME, SOLUTION, uniform, uniform, n_games=200)
    assert result["exploitability"] > 0.5 and result["best response"]["losses"] > result["best response"]["wins"]
    assert counts(result["nash"]) == "{wins}/{losses}/{ties}".format(**result["nash"])
