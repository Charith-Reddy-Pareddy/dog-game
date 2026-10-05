import numpy as np

from doggame.soccer import MIRROR_ACTION, Soccer
from doggame.soccer_solver import solve_soccer


def solved(width=5, height=3, goal_size=1):
    game = Soccer(width, height, goal_size)
    return game, solve_soccer(game)


def test_values_satisfy_the_bellman_equation():
    game, sol = solved()
    value = np.append(sol.value, 0.0)
    q = game.reward + 0.9 * value[game.next_state]
    assert np.allclose(sol.q, q, atol=1e-5)


def test_each_states_strategies_are_a_real_equilibrium():
    _, sol = solved()
    for s in range(len(sol.value)):
        value = sol.policy0[s] @ sol.q[s] @ sol.policy1[s]
        assert abs(value - sol.value[s]) < 1e-5
        assert (sol.policy0[s] @ sol.q[s]).min() >= value - 1e-5  # player 1 can't do better than playing it
        assert (sol.q[s] @ sol.policy1[s]).max() <= value + 1e-5  # player 0 can't do better


def test_the_game_is_fair_so_mirrored_states_have_opposite_values():
    game, sol = solved()
    assert np.allclose(sol.value[game.mirror()], -sol.value, atol=1e-5)


def test_starting_states_are_draws():
    game, sol = solved()
    assert np.allclose(sol.value[game.start_states()], 0.0, atol=1e-5)


def test_a_ball_holder_at_the_goal_mouth_has_already_won():
    game, sol = solved()
    s = game.index[((1, 3), (0, 0), 0)]  # player 0 one step from the goal opening, defender far away
    assert sol.value[s] > 0.8


def test_a_state_with_no_saddle_point_gets_a_mixed_strategy():
    from doggame.soccer_solver import _matrix_game

    q = np.zeros((1, 4, 4))
    q[0, :2, :2] = [[1.0, -1.0], [-1.0, 1.0]]  # matching pennies on the first two moves
    q[0, 2:, :] = -2.0  # the other moves are strictly worse for player 0

    value, policy0, policy1, mixed = _matrix_game(q)

    assert mixed[0]
    assert np.allclose(policy0[0], [0.5, 0.5, 0, 0], atol=1e-6)
    assert np.allclose(policy1[0], [0.5, 0.5, 0, 0], atol=1e-6)
    assert abs(value[0]) < 1e-6
