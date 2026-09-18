"""Stage-game Nash equilibrium solver for the dog game.

In the one-shot ("stage") version of the game, each player picks a
point to pull the dog toward their own house. Solving each player's
unconstrained best response and substituting one into the other shows
there is no interior fixed point unless the two houses share a
coordinate -- the unconstrained best-response recursion just keeps
pushing the action further in one direction. The bounded domain is
what turns that into a real equilibrium: players get pushed to the
boundary (they "exaggerate"), and iterated best response converges
there quickly.
"""

import numpy as np


def best_response_red(action_blue, house_red, w, domain):
    unconstrained = (house_red - (1 - w) * np.asarray(action_blue, dtype=float)) / w
    return domain.clip(unconstrained)


def best_response_blue(action_red, house_blue, w, domain):
    unconstrained = (house_blue - w * np.asarray(action_red, dtype=float)) / w
    return domain.clip(unconstrained)


def solve_stage_nash(house_red, house_blue, w, domain, tol=1e-10, max_iter=10_000):
    """Find a pure-strategy Nash equilibrium of the stage game via
    iterated best response.

    Returns (action_red, action_blue, dog) at the fixed point.
    """
    house_red = np.asarray(house_red, dtype=float)
    house_blue = np.asarray(house_blue, dtype=float)

    action_red = domain.clip(house_red)
    action_blue = domain.clip(house_blue)

    for _ in range(max_iter):
        new_red = best_response_red(action_blue, house_red, w, domain)
        new_blue = best_response_blue(new_red, house_blue, w, domain)
        converged = (
            np.max(np.abs(new_red - action_red)) < tol
            and np.max(np.abs(new_blue - action_blue)) < tol
        )
        action_red, action_blue = new_red, new_blue
        if converged:
            break

    dog = domain.clip(w * action_red + (1 - w) * action_blue)
    return action_red, action_blue, dog
