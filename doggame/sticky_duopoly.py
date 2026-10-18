"""A sticky-price duopoly in discrete time: an economic game for the learning results.

Two firms choose outputs q_i in [0, q_max]. The price moves part of the way toward the inverse demand,
p' = w0 p + (1 - w0)(A - q_1 - q_2), as in the sticky-price differential game of Fershtman and Kamien
(1987, Econometrica) and its discrete-time version (Heston and Hu, 2025, Operations Research), and each
firm earns p' q_i - c q_i - q_i^2 / 2, discounted. With w0 = 0 this is a static Cournot game with output
(A - c) / 4 each. With w0 > 0 a firm's output also moves future prices, so the feedback (Markov-perfect)
equilibrium differs from repeating the Cournot output.

``linear_equilibrium`` computes the symmetric linear feedback equilibrium exactly. (Nash value iteration on
a price grid, as doggame.inertia_dp does for the dog game, cycles here: grid equilibria jump between
neighbouring outputs from sweep to sweep.)
"""

import numpy as np

A, C, Q_MAX = 1.0, 0.1, 0.5


def cournot():
    """The static Cournot output and price."""
    q = (A - C) / 4
    return q, A - 2 * q


def next_price(p, q1, q2, w0):
    return np.clip(w0 * p + (1 - w0) * (A - q1 - q2), 0.0, A)


def profit(p_next, q):
    return p_next * q - C * q - q ** 2 / 2


def linear_equilibrium(w0, discount=0.9, tol=1e-13, max_iter=100000):
    """The symmetric linear feedback equilibrium q(p) = a + b p, exactly (no grid, no output bounds).

    With the rival playing a + b p and values V(p) = k0 + k1 p + k2 p^2, the next price is
    p' = m - s q with s = 1 - w0 and m = w0 p + s (A - a - b p), and a firm's one-round objective
    p' q - C q - q^2/2 + discount V(p') is a concave quadratic in its own q, maximized at
    q = (m (1 - 2 discount s k2) - C - discount s k1) / (1 + 2 s - 2 discount s^2 k2), linear in p.
    Iterating that best response and the value recursion to a fixed point gives the equilibrium.
    Returns (a, b, (k0, k1, k2))."""
    s, g = 1 - w0, discount
    a, b, k = (A - C) / 4, 0.0, np.zeros(3)
    for _ in range(max_iter):
        den = 1 + 2 * s - 2 * g * s ** 2 * k[2]
        # m = w0 p + s (A - a - b p) = m0 + m1 p
        m0, m1 = s * (A - a), w0 - s * b
        lift = 1 - 2 * g * s * k[2]
        na, nb = (m0 * lift - C - g * s * k[1]) / den, m1 * lift / den
        e0, e1 = s * (A - 2 * na), w0 - 2 * s * nb  # p' = e0 + e1 p under the new strategies
        # V(p) = p' q - C q - q^2 / 2 + g V(p'), matched coefficient by coefficient in p
        nk = np.array([e0 * na - C * na - na ** 2 / 2 + g * (k[0] + k[1] * e0 + k[2] * e0 ** 2),
                       e0 * nb + e1 * na - C * nb - na * nb + g * (k[1] * e1 + 2 * k[2] * e0 * e1),
                       e1 * nb - nb ** 2 / 2 + g * k[2] * e1 ** 2])
        done = max(abs(na - a), abs(nb - b), np.abs(nk - k).max()) < tol
        a, b, k = na, nb, nk
        if done:
            break
    return a, b, tuple(k)


def linear_steady_state(w0, discount=0.9):
    """The price and output the linear equilibrium settles at: p = w0 p + (1 - w0)(A - 2 (a + b p))."""
    a, b, _ = linear_equilibrium(w0, discount)
    p = (A - 2 * a) / (1 + 2 * b)
    return p, a + b * p


def deviation_gain(policy, w0, discount=0.9, prices=None, grid=801, rounds=300):
    """The most firm 1 gains, at any of the prices, by changing its output for one round and then
    following ``policy`` again, the rival following ``policy`` throughout (both firms play the same,
    symmetric, policy p -> q). Continuation profits come from simulating the price."""
    prices = np.linspace(0.05, 0.95, 19) if prices is None else prices
    best = 0.0
    for p in prices:
        q_now = float(policy(np.array([p]))[0])
        own = np.append(np.linspace(0, Q_MAX, grid), q_now)
        x = next_price(p, own, q_now, w0)
        total = profit(x, own)
        for k in range(1, rounds):
            q = policy(x)
            nx = next_price(x, q, q, w0)
            total += discount ** k * profit(nx, q)
            x = nx
        best = max(best, float(total[:-1].max() - total[-1]))
    return best
