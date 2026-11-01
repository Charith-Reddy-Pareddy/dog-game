"""Is the dog game a Markov potential game?

In a Markov potential game every player's change in value from switching its own policy equals the change in one
shared potential. Then the sum of the deviating players' value changes around any closed 4-cycle of policy profiles,

    (red, blue) -> (red', blue) -> (red', blue') -> (red, blue') -> (red, blue),

is exactly zero (it telescopes). A single cycle with a nonzero sum therefore proves the game is not one. The dynamics are
deterministic, so values are computed exactly by following the dog (the tail after `rounds` is below discount**rounds).
"""

import numpy as np


def value(red, blue, houses, w, w0, start, discount=0.9, rounds=400):
    """Both players' discounted rewards from `start` when red plays red(x) and blue plays blue(x) at dog position x."""
    house_red, house_blue = (np.asarray(h, dtype=float) for h in houses)
    x, totals = np.asarray(start, dtype=float), np.zeros(2)
    for t in range(rounds):
        x = w0 * x + (1 - w0) * (w * np.asarray(red(x)) + (1 - w) * np.asarray(blue(x)))
        totals -= discount ** t * np.array([np.sum((x - house_red) ** 2), np.sum((x - house_blue) ** 2)])
    return totals


def cycle_sum(red, red_alt, blue, blue_alt, houses, w, w0, start, discount=0.9):
    """The sum of the deviating players' value changes around the 4-cycle; zero in every Markov potential game."""
    v = lambda r, b: value(r, b, houses, w, w0, start, discount)
    return float((v(red_alt, blue)[0] - v(red, blue)[0]) + (v(red_alt, blue_alt)[1] - v(red_alt, blue)[1])
                 + (v(red, blue_alt)[0] - v(red_alt, blue_alt)[0]) + (v(red, blue)[1] - v(red, blue_alt)[1]))
