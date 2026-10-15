"""Is the one-shot Nash profile an equilibrium of the main dog game when the dog has inertia?

With dog inertia `w0` the dog keeps part of its old position, dog' = w0 * dog + (1 - w0) * (w * red + (1 - w) * blue),
so the game has memory. Both players hold their one-shot Nash action every round. If one player does, the other faces a
known, constant opponent, so its best response is an optimal-control problem over its own sequence of actions (any point in
the square each round). The profile is a Nash equilibrium of the dynamic game when neither player's best response beats
holding its action. This checks against every deviation by one player; the full feedback equilibrium is not computed.
"""

import numpy as np
from scipy.optimize import minimize


def discounted_reward(actions, other_action, own_weight, w0, start_dog, house, discount):
    """Sum over rounds of discount**t times minus the squared distance from the dog to `house`."""
    dog, total = np.asarray(start_dog, dtype=float), 0.0
    for t, action in enumerate(actions):
        dog = w0 * dog + (1 - w0) * (own_weight * action + (1 - own_weight) * other_action)
        total -= discount ** t * np.sum((dog - house) ** 2)
    return float(total)


def check_profile(houses, w, w0, actions, steps=20, discount=0.9):
    """For each player: the reward of holding its Nash action, its best response's reward and the gain from deviating.
    The dog starts where the environment puts it, between the two houses."""
    house_red, house_blue = (np.asarray(h, dtype=float) for h in houses)
    action_red, action_blue = (np.asarray(a, dtype=float) for a in actions)
    start = w * house_red + (1 - w) * house_blue
    result = {}
    for name, house, own, other, weight in (("red", house_red, action_red, action_blue, w), ("blue", house_blue, action_blue, action_red, 1 - w)):
        followed = discounted_reward(np.tile(own, (steps, 1)), other, weight, w0, start, house, discount)
        # With the opponent fixed, the state is affine in this player's action
        # sequence and the objective is a convex quadratic cost. L-BFGS-B is
        # therefore solving a box-constrained convex problem (not a local
        # nonconvex search). Supplying the exact adjoint gradient avoids the
        # finite-difference noise that hid the negative control.
        powers = discount ** np.arange(steps)

        def cost_and_gradient(flat):
            sequence = flat.reshape(steps, 2)
            dog, states = start.copy(), []
            for action in sequence:
                dog = w0 * dog + (1 - w0) * (weight * action + (1 - weight) * other)
                states.append(dog.copy())
            errors = np.asarray(states) - house
            cost = float(np.sum(powers[:, None] * errors**2))
            gradient = np.zeros_like(sequence)
            adjoint = np.zeros(2)
            for t in range(steps - 1, -1, -1):
                adjoint = 2 * powers[t] * errors[t] + w0 * adjoint
                gradient[t] = (1 - w0) * weight * adjoint
            return cost, gradient.ravel()

        initial = np.tile(own, (steps, 1)).ravel()
        found = minimize(cost_and_gradient, initial, method="L-BFGS-B", jac=True,
                         bounds=[(0.0, 1.0)] * (2 * steps),
                         options={"maxiter": 10000, "maxls": 50, "ftol": 1e-15, "gtol": 1e-10})
        if not found.success or not np.all(np.isfinite(found.x)):
            raise RuntimeError(f"{name} best-response optimization failed: {found.message}")
        best = max(-float(found.fun), followed)  # following is itself feasible; guard only round-off.
        result[name] = dict(followed=followed, best=best, gain=max(best - followed, 0.0))
    return result
