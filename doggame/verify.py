"""Nash-equilibrium verification.

The meeting described the actual check to use: "when I have a mixed
strategy, I can check if it's a Nash by computing the value for the
actions it's using with positive probability, and all of them should
be the same" -- plus, implicitly, that no unplayed action should beat
them. Two versions of that idea live here:

1. `exploitability` -- for a continuous action pair (e.g. a
   policy-gradient agent's action means, or the analytical solver's
   output), grid-search for the largest one-sided improvement either
   player could get by deviating. Zero (up to grid resolution) means
   neither player can do better alone, which *is* the definition of a
   Nash equilibrium.
2. `mixed_strategy_indifference_gap` -- for a mixed strategy over a
   discrete action grid (as fictitious play / DQN produce), directly
   the indifference check described above: the spread across values
   of actions played with positive probability, and the gap to the
   best unplayed action.
"""

import numpy as np

from doggame.env import negative_squared_distance_reward


def exploitability(action_red, action_blue, house_red, house_blue, w, domain, resolution=101):
    """Largest one-sided improvement available from
    (action_red, action_blue). Returns (red_gap, blue_gap); both are
    ~0 at a Nash equilibrium, up to grid resolution.
    """
    axis = np.linspace(domain.low, domain.high, resolution)
    xx, yy = np.meshgrid(axis, axis)
    candidates = np.stack([xx.ravel(), yy.ravel()], axis=1)

    action_red = np.asarray(action_red, dtype=float)
    action_blue = np.asarray(action_blue, dtype=float)

    dog = domain.clip(w * action_red + (1 - w) * action_blue)
    r_red, r_blue = negative_squared_distance_reward(dog, house_red, house_blue)

    alt_dog_red = domain.clip(w * candidates + (1 - w) * action_blue)
    alt_r_red = -np.sum((alt_dog_red - house_red) ** 2, axis=1)
    red_gap = max(0.0, float(alt_r_red.max()) - r_red)

    alt_dog_blue = domain.clip(w * action_red + (1 - w) * candidates)
    alt_r_blue = -np.sum((alt_dog_blue - house_blue) ** 2, axis=1)
    blue_gap = max(0.0, float(alt_r_blue.max()) - r_blue)

    return red_gap, blue_gap


def mixed_strategy_indifference_gap(payoff, strategy_row, strategy_opponent, tol=1e-3):
    """Check the classic mixed-Nash indifference condition for the row
    player: actions played with positive probability should all have
    equal value against the opponent's strategy, and no unplayed
    action should do strictly better.

    Returns (indifference_spread, best_response_gap); both are ~0 at
    a Nash equilibrium.

    `tol` is a probability floor for "played", not just >0: fictitious
    play's running average never fully forgets its random initial
    pick, so it leaves a permanent ~1/iterations residual on one
    action that isn't a real equilibrium violation. The default is
    tuned to that noise floor, not to genuine mixed strategies (whose
    components are typically far above 1%).

    On a fine enough action grid (more than 1/tol actions), a
    genuinely uniform mixed strategy spreads every action's
    probability below the floor, so nothing clears it. Falling back
    to the action(s) holding the strategy's actual maximum probability
    keeps `played` non-empty in that case, instead of leaving
    `best_response_gap` at +inf.
    """
    expected_payoffs = payoff @ strategy_opponent
    played = strategy_row > tol
    if not played.any():
        played = strategy_row == strategy_row.max()
    played_values = expected_payoffs[played]

    indifference_spread = float(played_values.max() - played_values.min())
    best_available = float(expected_payoffs.max())
    best_response_gap = max(0.0, best_available - float(played_values.max()))

    return indifference_spread, best_response_gap
