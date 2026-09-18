"""Turn the continuous action domain into a finite grid for DQN.

DQN needs a finite action set, so the square domain gets discretized
into an N x N grid of candidate points, and the stage-game payoff
matrices are built by calling the *same* transition/reward functions
the continuous environment uses (see doggame.env) -- so the grid game
and the continuous game are guaranteed to agree on the game's rules,
not just approximate them independently.
"""

import numpy as np


def action_grid(domain, resolution):
    """An (resolution**2, 2) array of grid points covering the domain."""
    axis = np.linspace(domain.low, domain.high, resolution)
    xx, yy = np.meshgrid(axis, axis)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def payoff_matrices(grid, house_red, house_blue, w, domain, transition_fn, reward_fn):
    """Build both players' payoff matrices over the discretized stage
    game: matrix[i, j] is the reward when red plays grid[i] and blue
    plays grid[j]."""
    n = len(grid)
    payoff_red = np.zeros((n, n))
    payoff_blue = np.zeros((n, n))
    for i, a_red in enumerate(grid):
        for j, a_blue in enumerate(grid):
            next_dog = domain.clip(transition_fn(None, a_red, a_blue, w))
            r_red, r_blue = reward_fn(next_dog, house_red, house_blue)
            payoff_red[i, j] = r_red
            payoff_blue[i, j] = r_blue
    return payoff_red, payoff_blue
