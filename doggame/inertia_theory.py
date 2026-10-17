"""A closed form for when holding the one-shot Nash actions is a Markov-perfect equilibrium of the game with inertia.

Work in one coordinate (the game splits by coordinate). Both players hold actions that put the dog at d*, so from dog
position x the dog moves to w0 x + (1 - w0) d*. By the one-shot deviation principle the profile is Markov-perfect
exactly when, at every x, no player gains by changing its action for one round. That round's payoff is a concave
quadratic in the next dog position z (see doggame.inertia_mpe), whose slope at the held z is, up to a positive factor,

    D(x) = (h - d*) / (1 - g w0)  -  w0 (x - d*) / (1 - g w0^2)

for a player with house h and discount g. A player that pushes the dog up as far as it can (its action at the top of
the box) holds optimally iff D(x) >= 0, one at the bottom iff D(x) <= 0, and one strictly inside iff D(x) = 0. D is
linear in x, so each player holds optimally on an interval of dog positions:

    top:     x <= d* + (h - d*) k / w0        bottom:  x >= d* - (d* - h) k / w0        inside:  x = d*

with k = (1 - g w0^2) / (1 - g w0) >= 1. Holding is Markov-perfect iff every interval covers [0, 1]; for a player
at the top that is (h - d*) k >= w0 (1 - d*). An inside action (where h = d*) fails at every other x, because the
player would lean against the dog's momentum.

Where holding fails, `explicit_policy` is a feedback equilibrium when the houses straddle the dog (each player at the
edge of the box nearest its house): each player aims the next dog position at its target d* + k (h - d*), as far as its
own weight allows. Proof outline (in full in docs/INERTIA_PROOFS.md): on the joint hold interval H the dog stays in H, so a player's one-round objective
g(z) = -(z - h)^2 + g_cont(z) is the concave quadratic above, peaked at the target, which lies in H. Outside H it
satisfies g(z) = -(z - h)^2 + g(next(z)) with next(z) nondecreasing and closer to the target, so the monotone shape on
each side of the target is preserved by a contraction and g is single-peaked. The best one-round reply over any
interval of reachable z is then the clipped target, which is the policy. It is checked numerically by
`deviation_gain`.
"""

import numpy as np


def kappa(w0, discount):
    return (1 - discount * w0 ** 2) / (1 - discount * w0)


def hold_interval(house, d_star, action, w0, discount, tol=1e-9):
    """Dog positions (one coordinate) where holding `action` is a best one-round response, as (low, high)."""
    if w0 == 0:
        return 0.0, 1.0
    reach = (house - d_star) * kappa(w0, discount) / w0
    if action >= 1 - tol:
        return 0.0, float(min(1.0, d_star + reach))
    if action <= tol:
        return float(max(0.0, d_star + reach)), 1.0
    return float(d_star), float(d_star)


def hold_share(houses, w, w0, actions, discount=0.9):
    """Per coordinate, the share of dog positions in [0, 1] where some player would rather not hold."""
    (h_red, h_blue), (a_red, a_blue) = ((np.asarray(v, dtype=float) for v in pair) for pair in (houses, actions))
    shares = []
    for c in range(len(h_red)):
        d_star = w * a_red[c] + (1 - w) * a_blue[c]
        lows, highs = zip(hold_interval(h_red[c], d_star, a_red[c], w0, discount), hold_interval(h_blue[c], d_star, a_blue[c], w0, discount))
        shares.append(float(1 - max(0.0, min(highs) - max(lows))))
    return shares


def holds_everywhere(houses, w, w0, actions, discount=0.9):
    """True iff holding the actions is a Markov-perfect equilibrium of the game with inertia."""
    return all(s < 1e-12 for s in hold_share(houses, w, w0, actions, discount))


def nash_1d(h_red, h_blue, w):
    """The one-shot Nash actions of one coordinate and their kind: "straddle" when the houses lie on either side of
    the dog both players' corners give (each player at the edge nearest its house), "interior" when one player stops
    short of its edge, and "shared" when the houses coincide (a continuum; the profile returned is one member)."""
    if h_red == h_blue:
        return 1.0, (h_red - w) / (1 - w) if h_red >= w else 0.0, "shared"
    if h_red < h_blue:  # mirror the coordinate so red's house is the higher one
        a_red, a_blue, kind = nash_1d(1 - h_red, 1 - h_blue, w)
        return 1 - a_red, 1 - a_blue, kind
    if h_blue <= w <= h_red:
        return 1.0, 0.0, "straddle"
    return (h_red / w, 0.0, "interior") if w > h_red else (1.0, (h_blue - w) / (1 - w), "interior")


def target(house, d_star, w0, discount):
    """Where a player would put the next dog position for one round, if both hold afterwards: d* + k (h - d*)."""
    return d_star + (house - d_star) * kappa(w0, discount)


def explicit_policy(h_red, h_blue, w, a_red, a_blue, w0, discount=0.9):
    """A feedback policy for one coordinate: each player aims the next dog position at its target, as far as its own
    weight allows, with the other at its one-shot Nash action. It is the one-shot profile on the hold intervals and
    piecewise affine in x outside them. Returns x -> (red action, blue action)."""
    d_star = w * a_red + (1 - w) * a_blue
    t_red, t_blue = target(h_red, d_star, w0, discount), target(h_blue, d_star, w0, discount)

    def act(x):
        held = w0 * np.asarray(x, dtype=float) + (1 - w0) * d_star
        red = np.clip(a_red + (t_red - held) / ((1 - w0) * w), 0, 1)
        blue = np.clip(a_blue + (t_blue - held) / ((1 - w0) * (1 - w)), 0, 1)
        return red, blue
    return act


def deviation_gain(act, house, w, w0, x, red=True, discount=0.9, rounds=300, grid=2001):
    """The most one player gains at dog position x (one coordinate) by changing its action for one round and then
    following `act` again. Continuation values come from simulating the deterministic dog, not from a state grid."""
    a_red, a_blue = act(x)
    own = np.append(np.linspace(0, 1, grid), a_red if red else a_blue)
    z = w0 * x + (1 - w0) * (w * own + (1 - w) * a_blue if red else w * a_red + (1 - w) * own)
    total = -(z - house) ** 2
    for k in range(1, rounds):
        r, b = act(z)
        z = w0 * z + (1 - w0) * (w * r + (1 - w) * b)
        total -= discount ** k * (z - house) ** 2
    return float(max(total[:-1].max() - total[-1], 0.0))
