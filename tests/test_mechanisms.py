import numpy as np

from doggame import many_houses as H
from doggame import mechanisms as M

HOUSES, WEIGHTS = np.array([0.8, 0.45, 0.1]), np.full(3, 1 / 3)
SMALL = dict(states=6, grid=51, rounds=80)


def test_the_median_rule_moves_the_outcome_to_the_median_target_as_far_as_it_can():
    x = np.array([0.0, 0.5, 1.0])
    assert np.allclose(M.next_median(x, [0.9, 0.4, 0.1], 0.6), np.clip(0.4, 0.6 * x, 0.6 * x + 0.4))


def test_the_mean_rule_targets_match_the_many_house_solver():
    targets, _ = M.target_equilibrium("mean", HOUSES, WEIGHTS, 0.6)
    assert np.allclose(targets, H.target_equilibrium(HOUSES, WEIGHTS, 0.6)[0], atol=2e-3)


def test_truthful_median_reporting_survives_without_inertia_but_not_with_it():
    truthful = lambda x: HOUSES.copy()
    assert M.deviation_gain("median", truthful, HOUSES, WEIGHTS, 0.0, **SMALL) < 1e-12
    assert M.deviation_gain("median", truthful, HOUSES, WEIGHTS, 0.6, **SMALL) > 0.01


def test_the_target_form_is_an_equilibrium_under_both_rules_and_the_planner_loses_least():
    planner = M.social_loss(lambda x: M.planner_step(x, HOUSES, 0.6), HOUSES)
    for rule in M.RULES:
        targets, _ = M.target_equilibrium(rule, HOUSES, WEIGHTS, 0.6)
        policy = lambda v, rule=rule, targets=targets: M.reports(rule, v, targets, WEIGHTS, 0.6)
        assert M.deviation_gain(rule, policy, HOUSES, WEIGHTS, 0.6, **SMALL) < 1e-6
        assert M.social_loss(lambda x: M.step(rule, x, targets, WEIGHTS, 0.6), HOUSES) >= planner
