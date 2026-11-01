"""The inertia modules checked against each other: the closed form (inertia_theory, inertia_mpe), the dynamic
program (inertia_dp) and the fixed-opponent check from the start (inertia_equilibrium). Each test is an implication
that must hold whatever the game."""
import numpy as np

from doggame.env import Square
from doggame.experiments import DEFAULT_CONFIGS
from doggame.inertia_dp import solve_1d
from doggame.inertia_mpe import value
from doggame.inertia_theory import hold_share
from doggame.nash import solve_stage_nash


def setup(index):
    c = DEFAULT_CONFIGS[index]
    houses = (np.array(c["house_red"]), np.array(c["house_blue"]))
    return houses, c["w"], tuple(solve_stage_nash(houses[0], houses[1], c["w"], Square(0.0, 1.0))[:2])


def test_the_dynamic_program_leaves_the_one_shot_actions_where_the_closed_form_says_someone_gains():
    for index in range(len(DEFAULT_CONFIGS)):
        houses, w, nash = setup(index)
        share = hold_share(houses, w, 0.6, nash)
        for c in range(2):
            if share[c] > 0.05:
                part = solve_1d(houses[0][c], houses[1][c], w, 0.6, states=41, actions=21)
                assert np.abs(part["red"] - nash[0][c]).max() > 1e-9 or np.abs(part["blue"] - nash[1][c]).max() > 1e-9


def test_where_the_dynamic_program_holds_the_one_shot_actions_its_values_converge_to_the_closed_form():
    houses, w, nash = setup(0)
    h_red, h_blue, d_star = houses[0][0], houses[1][0], w * nash[0][0] + (1 - w) * nash[1][0]
    errors = []
    for states in (41, 81):  # the program interpolates values linearly between states, so the error falls like spacing^2
        part = solve_1d(h_red, h_blue, w, 0.3, states=states, actions=21)
        assert np.allclose(part["red"], nash[0][0]) and np.allclose(part["blue"], nash[1][0])
        errors.append(np.abs(part["v_red"] - [value([x], [d_star], [h_red], 0.3, 0.9) for x in part["z"]]).max())
    assert errors[0] < 1e-4 and errors[1] < errors[0] / 3
