"""Best response over a continuous action, three ways.

Given a function `q(a)` (a player's Q-value as a function of their own
action, with the opponent's action held fixed) these find the `a` in
[low, high] that maximizes it:

- `bisect_best_response`: bisect on the sign of the slope.
- `gradient_best_response`: climb the slope, taking steps of finite-difference gradients.
- `quadratic_best_response`: fit a parabola to a few samples and jump to its peak.

All three assume `q` has a single peak on the interval.
"""

import numpy as np


def _slope(q, a, eps):
    return (q(a + eps) - q(a - eps)) / (2 * eps)


def bisect_best_response(q, low, high, eps=1e-5, tol=1e-6):
    while high - low > tol:
        mid = (low + high) / 2
        if _slope(q, mid, eps) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def gradient_best_response(q, low, high, start=None, lr=0.1, steps=500, eps=1e-5):
    a = (low + high) / 2 if start is None else start
    for _ in range(steps):
        a = float(np.clip(a + lr * _slope(q, a, eps), low, high))
    return a


def quadratic_best_response(q, low, high, samples=5):
    points = np.linspace(low, high, samples)
    values = np.array([q(a) for a in points])
    best = int(np.argmax(values))
    # fit a parabola through three samples around the best one (always
    # three, even at an edge, since a parabola needs three points)
    start = min(max(best - 1, 0), samples - 3)
    near = slice(start, start + 3)
    c2, c1, _ = np.polyfit(points[near], values[near], 2)
    if c2 >= 0:  # not curving down: the peak is at a sample (an edge)
        return float(points[best])
    return float(np.clip(-c1 / (2 * c2), low, high))
