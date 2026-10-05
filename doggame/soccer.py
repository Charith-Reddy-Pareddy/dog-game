"""A deterministic two-player grid soccer game (a zero-sum Markov game).

Both players pick one of four moves each turn, at the same time. Whoever
holds the ball scores by walking out through the opponent's goal opening
(player 0 attacks the right side, player 1 the left). A score gives the
scorer +1 and ends the game; the other player gets -1.

Collision rules, chosen so the game is the same for both players:
- a move off the board, or into a wall beside the goal, does nothing;
- if both players aim at the same cell, or try to swap cells, neither moves;
- if a player walks into an opponent who stays put, the walker doesn't
  move, and if the walker held the ball it goes to the opponent.

The whole transition is precomputed into tables so a solver can use it
directly. State index `n_states` is the finished game.
"""

import numpy as np

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # north, south, west, east
N_ACTIONS = len(MOVES)
MIRROR_ACTION = [0, 1, 3, 2]  # a mirrored board swaps west and east


class Soccer:
    def __init__(self, width=7, height=5, goal_size=3):
        self.width, self.height = width, height
        top = (height - goal_size) // 2
        self.goal_rows = range(top, top + goal_size)

        cells = [(r, c) for r in range(height) for c in range(width)]
        self.states = [(p0, p1, ball) for p0 in cells for p1 in cells if p0 != p1 for ball in (0, 1)]
        self.index = {state: i for i, state in enumerate(self.states)}
        self.n_states = len(self.states)
        self._build_tables()

    def _resolve(self, p, ball, actions):
        """One simultaneous turn. Returns (new positions, new ball, reward for player 0, finished)."""
        target = [(p[i][0] + MOVES[actions[i]][0], p[i][1] + MOVES[actions[i]][1]) for i in (0, 1)]

        row, col = target[ball]
        if row in self.goal_rows and col in (-1, self.width):
            return p, ball, (1.0 if col == self.width else -1.0), True

        for i in (0, 1):
            if not (0 <= target[i][0] < self.height and 0 <= target[i][1] < self.width):
                target[i] = p[i]

        walks_into_0 = target[1] == p[0] and target[0] == p[0]  # player 1 bumps a player 0 who stays
        walks_into_1 = target[0] == p[1] and target[1] == p[1]
        if walks_into_1:
            new = list(p)
            ball = 1 if ball == 0 else ball
        elif walks_into_0:
            new = list(p)
            ball = 0 if ball == 1 else ball
        elif target[0] == target[1] or (target[0] == p[1] and target[1] == p[0]):
            new = list(p)
        else:
            new = target
        return tuple(new), ball, 0.0, False

    def _build_tables(self):
        shape = (self.n_states, N_ACTIONS, N_ACTIONS)
        self.next_state = np.full(shape, self.n_states, dtype=np.int64)
        self.reward = np.zeros(shape)
        for s, (p0, p1, ball) in enumerate(self.states):
            for a0 in range(N_ACTIONS):
                for a1 in range(N_ACTIONS):
                    new, new_ball, reward, finished = self._resolve((p0, p1), ball, (a0, a1))
                    self.reward[s, a0, a1] = reward
                    if not finished:
                        self.next_state[s, a0, a1] = self.index[(new[0], new[1], new_ball)]

    def start_states(self):
        """Players face each other mid-board; either player may start with the ball."""
        mid = self.height // 2
        p0, p1 = (mid, 1), (mid, self.width - 2)
        return [self.index[(p0, p1, 0)], self.index[(p0, p1, 1)]]

    def features(self, states):
        """Positions scaled to [0, 1], plus who holds the ball: a (n, 5) array."""
        states = np.asarray(states)
        out = np.zeros((len(states), 5), dtype=np.float32)
        for k, s in enumerate(states):
            (r0, c0), (r1, c1), ball = self.states[s]
            out[k] = [r0 / (self.height - 1), c0 / (self.width - 1), r1 / (self.height - 1), c1 / (self.width - 1), ball]
        return out

    def mirror(self):
        """For every state, the mirror image: board flipped left-right and the
        players swapped. A fair game plays the same from a state and its mirror."""
        flip = lambda cell: (cell[0], self.width - 1 - cell[1])
        return np.array([self.index[(flip(p1), flip(p0), 1 - ball)] for p0, p1, ball in self.states])
