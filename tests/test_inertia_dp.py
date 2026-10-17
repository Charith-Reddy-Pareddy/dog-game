import numpy as np

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_dp import policy, solve, solve_1d
from doggame.inertia_mpe import check_mpe
from doggame.nash import solve_stage_nash

SMALL = dict(states=41, actions=21)


def setup(index):
    c = DEFAULT_CONFIGS[index]
    houses = (np.array(c["house_red"]), np.array(c["house_blue"]))
    return houses, c["w"], tuple(solve_stage_nash(houses[0], houses[1], c["w"], Square(0.0, 1.0))[:2])


def test_without_inertia_the_feedback_equilibrium_is_the_one_shot_nash_at_every_state():
    houses, w, (nash_red, nash_blue) = setup(0)
    for c, part in enumerate(solve(houses, w, 0.0, **SMALL)):
        assert np.allclose(part["red"], nash_red[c]) and np.allclose(part["blue"], nash_blue[c]) and part["unresolved"] == 0


def test_it_agrees_with_the_state_by_state_check_where_holding_nash_is_already_an_equilibrium():
    houses, w, nash = setup(0)
    assert check_mpe(houses, w, 0.3, nash, grid=11)["blue"]["max_gain"] < 1e-12  # holding Nash cannot be beaten here

    for c, part in enumerate(solve(houses, w, 0.3, **SMALL)):
        assert np.allclose(part["red"], nash[0][c]) and np.allclose(part["blue"], nash[1][c])


def test_off_the_path_it_eases_off_where_the_state_by_state_check_found_a_gain():
    houses, w, (nash_red, nash_blue) = setup(1)  # blue gains when the dog already sits on blue's corner
    solution = solve(houses, w, 0.6, **SMALL)

    red, blue = policy(solution, (0.0, 0.0))
    assert np.any(np.abs(blue - nash_blue) > 1e-9)
    red, blue = policy(solution, w * houses[0] + (1 - w) * houses[1])  # but from the starting position it plays Nash
    assert np.allclose(red, nash_red) and np.allclose(blue, nash_blue)


def test_value_iteration_converges_and_values_are_nonpositive():
    part = solve_1d(0.9, 0.1, 0.5, 0.6, **SMALL)

    assert part["sweeps"] < 2000 and np.all(part["v_red"] <= 0) and np.all(part["v_blue"] <= 0)


def test_a_noisy_dog_keeps_the_equilibrium_bang_bang_and_lowers_both_values():
    from doggame.inertia_dp import solve_1d

    calm = solve_1d(0.7, 0.6, 0.7, 0.9, discount=0.95, states=61, actions=11, tol=1e-8)
    noisy = solve_1d(0.7, 0.6, 0.7, 0.9, discount=0.95, states=61, actions=11, tol=1e-8, noise=0.05)
    for sol in (calm, noisy):
        assert np.mean((sol["red"] > 0) & (sol["red"] < 1)) < 0.1 and np.mean((sol["blue"] > 0) & (sol["blue"] < 1)) < 0.1
    assert noisy["v_red"].mean() < calm["v_red"].mean() and noisy["v_blue"].mean() < calm["v_blue"].mean()
