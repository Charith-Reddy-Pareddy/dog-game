"""Continuous-state, continuous-action environment for the dog game.

Two players ("red" and "blue") each choose a point in a bounded, convex
domain every round. The dog's next position is a convex combination of
those two points, weighted by `w`. Each player is rewarded for pulling
the dog close to their own house.

The transition and reward functions are injected rather than hardcoded,
so the same loop can later drive a different game (e.g. a soccer game)
without changing the environment class itself.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Square:
    """A closed, bounded [low, high] x [low, high] domain."""

    low: float = 0.0
    high: float = 1.0

    def clip(self, point):
        return np.clip(point, self.low, self.high)

    def sample(self, rng):
        return rng.uniform(self.low, self.high, size=2)


def weighted_average_transition(dog, action_red, action_blue, w):
    """Default transition: the dog moves to a w / (1 - w) blend of the
    two players' chosen points, independent of its previous position."""
    del dog
    return w * np.asarray(action_red, dtype=float) + (1 - w) * np.asarray(
        action_blue, dtype=float
    )


def negative_squared_distance_reward(next_dog, house_red, house_blue):
    """Default reward: each player is penalized by squared distance
    from the dog to their own house."""
    r_red = -float(np.sum((next_dog - house_blue) ** 2))
    r_blue = -float(np.sum((next_dog - house_red) ** 2))
    return r_red, r_blue


class DogGameEnv:
    """A repeated two-player game over the position of a shared dog."""

    def __init__(
        self,
        house_red,
        house_blue,
        w=0.5,
        domain=None,
        discount=0.95,
        transition_fn=weighted_average_transition,
        reward_fn=negative_squared_distance_reward,
        seed=None,
    ):
        if not 0.0 < w < 1.0:
            raise ValueError("w must be strictly between 0 and 1")
        self.house_red = np.asarray(house_red, dtype=float)
        self.house_blue = np.asarray(house_blue, dtype=float)
        self.w = w
        self.domain = domain or Square()
        self.discount = discount
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng(seed)
        self.dog = None

    def reset(self):
        start = self.w * self.house_red + (1 - self.w) * self.house_blue
        self.dog = self.domain.clip(start)
        return self.dog.copy()

    def step(self, action_red, action_blue):
        action_red = self.domain.clip(action_red)
        action_blue = self.domain.clip(action_blue)
        next_dog = self.transition_fn(self.dog, action_red, action_blue, self.w)
        next_dog = self.domain.clip(next_dog)
        r_red, r_blue = self.reward_fn(next_dog, self.house_red, self.house_blue)
        self.dog = next_dog
        return next_dog.copy(), (r_red, r_blue), False, {}
