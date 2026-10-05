import numpy as np

from doggame.soccer import MIRROR_ACTION, N_ACTIONS, Soccer

NORTH, SOUTH, WEST, EAST = range(4)


def small():
    return Soccer(width=5, height=3, goal_size=1)  # one goal cell, in the middle row


def step(game, p0, p1, ball, a0, a1):
    s = game.index[(p0, p1, ball)]
    nxt = game.next_state[s, a0, a1]
    return (game.states[nxt] if nxt < game.n_states else None), game.reward[s, a0, a1]


def test_state_count_is_all_distinct_cell_pairs_times_two_ball_holders():
    game = Soccer(width=7, height=5, goal_size=3)
    assert game.n_states == 35 * 34 * 2
    assert game.next_state.shape == (game.n_states, N_ACTIONS, N_ACTIONS)


def test_ball_holder_walking_out_the_opponents_goal_scores():
    game = small()
    new, reward = step(game, (1, 4), (0, 0), 0, EAST, NORTH)  # player 0 walks right out of the goal row
    assert new is None and reward == 1.0
    new, reward = step(game, (0, 4), (1, 0), 1, EAST, WEST)  # player 1 walks left out of the goal row
    assert new is None and reward == -1.0


def test_walking_into_your_own_goal_scores_for_the_opponent():
    game = small()
    _, reward = step(game, (1, 0), (0, 4), 0, WEST, SOUTH)
    assert reward == -1.0


def test_the_wall_beside_the_goal_blocks_the_ball_holder():
    game = small()
    new, reward = step(game, (0, 4), (2, 0), 0, EAST, EAST)  # row 0 is not a goal row
    assert reward == 0.0 and new[0] == (0, 4)


def test_players_aiming_at_the_same_cell_both_stay():
    game = small()
    new, _ = step(game, (1, 1), (1, 3), 0, EAST, WEST)
    assert new[0] == (1, 1) and new[1] == (1, 3)


def test_players_trying_to_swap_cells_both_stay():
    game = small()
    new, _ = step(game, (1, 1), (1, 2), 0, EAST, WEST)
    assert new[0] == (1, 1) and new[1] == (1, 2)


def test_walking_into_a_cell_the_opponent_is_leaving_just_works():
    game = small()
    new, _ = step(game, (1, 1), (1, 2), 0, EAST, NORTH)  # player 1 steps out of the way
    assert new[0] == (1, 2) and new[2] == 0


def test_bumping_a_player_who_stays_put_hands_over_the_ball():
    game = small()
    # player 1 stays (walks into the wall below), player 0 holds the ball and walks into them
    new, _ = step(game, (2, 1), (2, 2), 0, EAST, SOUTH)
    assert new[0] == (2, 1) and new[2] == 1


def test_the_game_is_the_same_for_both_players():
    game = small()
    mirror = game.mirror()
    swap = np.array(MIRROR_ACTION)
    for s in range(game.n_states):
        for a0 in range(N_ACTIONS):
            for a1 in range(N_ACTIONS):
                nxt = game.next_state[s, a0, a1]
                # player 1 seeing the mirror image acts like player 0 here
                mirrored = game.next_state[mirror[s], swap[a1], swap[a0]]
                assert (mirrored == game.n_states) == (nxt == game.n_states)
                if nxt < game.n_states:
                    assert mirrored == mirror[nxt]
                assert game.reward[mirror[s], swap[a1], swap[a0]] == -game.reward[s, a0, a1]
