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


def test_zero_shocks_reproduce_the_deterministic_targets_and_loss():
    zero = M.shocks(0.0, 0.6)
    for rule in M.RULES:
        targets, _ = M.target_equilibrium(rule, HOUSES, WEIGHTS, 0.6, grid=201)
        assert np.allclose(M.target_equilibrium(rule, HOUSES, WEIGHTS, 0.6, grid=201, shock=zero)[0], targets, atol=0.01)
        move = lambda x, rule=rule, targets=targets: M.step(rule, x, targets, WEIGHTS, 0.6)
        assert abs(M.social_loss(move, HOUSES, shock=zero) - M.social_loss(move, HOUSES)) < 1e-3


def test_inertia_filters_a_shock_that_comes_with_the_aggregate():
    assert np.allclose(M.shocks(0.2, 0.75)[0], 0.25 * M.shocks(0.2, 0.0)[0])
    assert np.allclose(M.shocks(0.2, 0.75, filtered=False)[0], M.shocks(0.2, 0.0)[0])


def test_with_shocks_the_optimal_planner_loses_no_more_than_the_greedy_one():
    for noise, w0 in ((0.4, 0.0), (0.2, 0.6)):
        shock = M.shocks(noise, w0)
        target = M.planner_target(HOUSES, w0, shock=shock)
        greedy = M.social_loss(lambda x: M.planner_step(x, HOUSES, w0), HOUSES, shock=shock)
        assert M.social_loss(lambda x: M.next_median(x, [target], w0), HOUSES, shock=shock) <= greedy + 1e-9


def test_the_greedy_planner_loss_falls_with_inertia_while_the_mean_house_stays_reachable():
    houses = np.array([0.2, 0.5, 0.6])
    bound, losses = min(houses.mean(), 1 - houses.mean()), []
    for w0 in (0.0, 0.2, 0.4):
        assert w0 <= bound
        shock = M.shocks(0.3, w0)
        nodes, probs = shock
        spread = ((np.clip(houses.mean() + nodes, 0, 1) - houses.mean()) ** 2) @ probs
        losses.append(M.social_loss(lambda x: M.planner_step(x, houses, w0), houses, shock=shock))
        assert np.isclose(losses[-1], (((houses - houses.mean()) ** 2).sum() + 3 * spread) / 0.1)
    assert losses[0] > losses[1] > losses[2]


def test_in_the_continuous_limit_the_median_rule_runs_and_the_mean_rule_is_a_tug_of_war():
    from scipy.integrate import solve_ivp

    houses, weights, dt, rate = np.array([0.05, 0.2, 0.45, 0.7, 0.95]), np.full(5, 0.2), 0.02, 0.5
    w0, discount = 1 - dt, np.exp(-rate * dt)
    times = np.arange(int(4 / dt) + 1) * dt
    for rule in M.RULES:
        targets, _ = M.target_equilibrium(rule, houses, weights, w0, discount, grid=1601, rounds=1400)
        x, path = np.array([0.0]), [0.0]
        for _ in times[1:]:
            x = M.step(rule, x, targets, weights, w0)
            path.append(float(x[0]))
        if rule == "median":  # full speed, dx/dt = 1 - x, until the median house
            limit = np.minimum(1 - np.exp(-times), np.median(houses))
        else:  # dx/dt = U(x) - x, U the weight of players whose targets lie above x
            ode = solve_ivp(lambda _, y: [weights[targets > y[0]].sum() - y[0]], (0, times[-1]), [0.0], t_eval=times, max_step=dt / 4)
            limit = ode.y[0]
        assert np.abs(np.array(path) - limit).max() < 0.005
