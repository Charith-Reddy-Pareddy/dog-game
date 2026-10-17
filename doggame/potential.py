"""The one-shot dog game is an exact potential game.

With the dog at d = w * a_red + (1 - w) * a_blue and rewards u_i = -||d - h_i||^2,

    Phi(a_red, a_blue) = -||d||^2 + 2 w h_red . a_red + 2 (1 - w) h_blue . a_blue

changes by exactly the deviating player's reward change under any one-player
deviation (the cross-partials of u_red and u_blue are both -2 w (1 - w) I).
So every maximizer of Phi over the square is a Nash equilibrium.

Moving both players so the dog stays put, a_red += (1 - w) * u and
a_blue -= w * u, changes Phi by 2 w (1 - w) (h_red - h_blue) . u. Phi therefore
rises toward the edge in every coordinate where the houses differ, and is flat
in a coordinate the houses share: that flat direction is the continuum of
equilibria found when both houses have the same y.
"""

import numpy as np


def potential(a_red, a_blue, house_red, house_blue, w):
    a_red, a_blue = np.asarray(a_red, dtype=float), np.asarray(a_blue, dtype=float)
    dog = w * a_red + (1 - w) * a_blue
    return float(-dog @ dog + 2 * w * np.dot(house_red, a_red) + 2 * (1 - w) * np.dot(house_blue, a_blue))


def rewards(a_red, a_blue, house_red, house_blue, w):
    dog = w * np.asarray(a_red, dtype=float) + (1 - w) * np.asarray(a_blue, dtype=float)
    return float(-np.sum((dog - house_red) ** 2)), float(-np.sum((dog - house_blue) ** 2))


def slope_along_flat_move(house_red, house_blue, w):
    """How fast Phi changes per unit of the dog-preserving move, per coordinate."""
    return 2 * w * (1 - w) * (np.asarray(house_red, dtype=float) - np.asarray(house_blue, dtype=float))
