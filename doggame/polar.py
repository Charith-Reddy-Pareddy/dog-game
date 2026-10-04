"""Dog game where the players walk, using angle-radius actions.

Each player stands somewhere in the square and every round picks a move
(theta, r): head in direction theta and walk a distance r, up to
`max_step`. The dog then goes to a blend of the dog's old position and
the two players' new positions (`w0 * dog + w1 * red + w2 * blue`, the
same blend `make_inertial_transition` builds).

This is one reading of "angle-radius output" -- see ASSUMPTIONS.md.
"""

import numpy as np

from doggame.env import (
    Square,
    make_inertial_transition,
    negative_squared_distance_reward,
    weighted_average_transition,
)


class PolarDogGameEnv:
    def __init__(self, house_red, house_blue, w=0.5, w0=0.0, max_step=0.2, domain=None, discount=0.9):
        if not 0.0 < w < 1.0:
            raise ValueError("w must be strictly between 0 and 1")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.house_red = np.asarray(house_red, dtype=float)
        self.house_blue = np.asarray(house_blue, dtype=float)
        self.w = w
        self.max_step = max_step
        self.domain = domain or Square()
        self.discount = discount
        self.transition = make_inertial_transition(w0)

    def reset(self):
        """Players start on their own houses, the dog between them."""
        self.red = self.domain.clip(self.house_red)
        self.blue = self.domain.clip(self.house_blue)
        self.dog = weighted_average_transition(None, self.red, self.blue, self.w)
        return self.state()

    def state(self):
        return np.concatenate([self.dog, self.red, self.blue])

    def _walk(self, position, theta, r):
        r = np.clip(r, 0.0, self.max_step)
        step = r * np.array([np.cos(theta), np.sin(theta)])
        return self.domain.clip(position + step)

    def step(self, move_red, move_blue):
        """Each move is (theta, r). Returns (state, (reward_red, reward_blue))."""
        self.red = self._walk(self.red, *move_red)
        self.blue = self._walk(self.blue, *move_blue)
        self.dog = self.domain.clip(self.transition(self.dog, self.red, self.blue, self.w))
        rewards = negative_squared_distance_reward(self.dog, self.house_red, self.house_blue)
        return self.state(), rewards
