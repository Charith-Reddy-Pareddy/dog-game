import numpy as np

from doggame.soccer import Soccer
from doggame.soccer_exact_play import duality_gap, play, stage_equilibrium
from doggame.soccer_solver import solve_soccer


def test_the_solved_equilibrium_has_no_duality_gap_and_a_uniform_pair_does():
    game = Soccer(width=4, height=3, goal_size=1)
    solution = solve_soccer(game, discount=0.9, tol=1e-10)
    assert np.abs(duality_gap(solution.policy0, solution.policy1, game)).max() < 1e-5
    uniform = np.full((game.n_states, 4), 0.25)
    assert duality_gap(uniform, uniform, game).max() > 0.1


def test_exact_play_with_a_table_lowers_the_gap():
    rows = play(Soccer(width=4, height=3, goal_size=1), "table", steps=150, every=150)
    assert rows[-1][1] < 0.5 * rows[0][1]


def test_stage_equilibria_are_pure_saddle_points_or_the_minimax_mix():
    saddle = np.array([[3.0, 1.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    pennies = np.zeros((4, 4))
    pennies[:2, :2] = [[1.0, -1.0], [-1.0, 1.0]]
    pennies[2:, :] = -5.0
    pennies[:2, 2:] = 5.0
    p0, p1 = stage_equilibrium(np.stack([saddle, pennies]))
    assert np.allclose(p0[0], [1, 0, 0, 0]) and np.allclose(p1[0], [0, 1, 0, 0])
    assert np.allclose(p0[1], [0.5, 0.5, 0, 0], atol=1e-6) and np.allclose(p1[1], [0.5, 0.5, 0, 0], atol=1e-6)
