"""Ways to judge a soccer policy. A policy is a table: `policy[s]` is the
probability of each of the four moves in state `s`.

- `play_games`: play it out many times and count wins, losses and ties.
  A game that reaches the step limit is a tie.
- `best_response`: the best possible opponent, found exactly.
- `exploitability`: how much two policies together leave on the table.
- `symmetry_gap`: how far the two players' policies are from mirroring
  each other, which a fair game's equilibrium should do.
"""

import numpy as np

from doggame.soccer import MIRROR_ACTION, N_ACTIONS


def random_policy(game):
    return np.full((game.n_states, N_ACTIONS), 1.0 / N_ACTIONS)


def play_games(game, policy0, policy1, n_games=1000, max_steps=100, seed=0):
    """Wins, losses and ties for player 0. Games alternate between the two starting states."""
    rng = np.random.default_rng(seed)
    starts = game.start_states()
    state = np.array([starts[i % 2] for i in range(n_games)])
    result = np.zeros(n_games)  # +1 win, -1 loss, 0 tie
    active = np.ones(n_games, dtype=bool)

    def sample(policy, states):
        return (rng.random(len(states))[:, None] > policy[states].cumsum(axis=1)).sum(axis=1).clip(max=N_ACTIONS - 1)

    for _ in range(max_steps):
        live = np.flatnonzero(active)
        if len(live) == 0:
            break
        a0, a1 = sample(policy0, state[live]), sample(policy1, state[live])
        reward = game.reward[state[live], a0, a1]
        nxt = game.next_state[state[live], a0, a1]
        result[live] = reward
        state[live] = np.minimum(nxt, game.n_states - 1)
        active[live[nxt == game.n_states]] = False
    return {"wins": int((result > 0).sum()), "losses": int((result < 0).sum()), "ties": int((result == 0).sum())}


def best_response(game, opponent, player, discount=0.9, iterations=300):
    """The best policy for `player` against a fixed `opponent`, and its value
    in every state (from player 0's point of view)."""
    value = np.zeros(game.n_states + 1)
    for _ in range(iterations):
        future = game.reward + discount * value[game.next_state]  # (state, move0, move1)
        if player == 0:
            q = np.einsum("sab,sb->sa", future, opponent)
            new_value = q.max(axis=1)
        else:
            q = np.einsum("sab,sa->sb", future, opponent)
            new_value = q.min(axis=1)
        if np.abs(new_value - value[:-1]).max() < 1e-9:
            break
        value[:-1] = new_value
    best = q.argmax(axis=1) if player == 0 else q.argmin(axis=1)
    return np.eye(N_ACTIONS)[best], value[:-1]


def exploitability(game, policy0, policy1, discount=0.9):
    """At the starting states: what player 0 could win against `policy1` plus
    what player 1 could win against `policy0`. Zero only at an equilibrium."""
    _, value0 = best_response(game, policy1, player=0, discount=discount)
    _, value1 = best_response(game, policy0, player=1, discount=discount)
    starts = game.start_states()
    return float((value0[starts] - value1[starts]).mean())


def symmetry_gap(game, policy0, policy1):
    """Average difference between player 0's policy and player 1's policy at
    the mirror-image state (with west and east swapped)."""
    mirrored = policy1[game.mirror()][:, MIRROR_ACTION]
    return float(np.abs(policy0 - mirrored).sum(axis=1).mean())
