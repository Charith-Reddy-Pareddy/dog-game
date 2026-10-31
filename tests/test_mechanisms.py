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


def test_the_mean_rule_is_the_generalized_median_with_evenly_spread_phantoms_and_the_median_rule_with_edge_phantoms():
    houses, weights = np.array([0.15, 0.3, 0.9, 0.55, 0.05]), np.full(5, 0.2)
    for rule, phantoms in (("mean", (0.2, 0.4, 0.6, 0.8)), ("median", (0.0, 0.0, 1.0, 1.0))):
        targets, _ = M.target_equilibrium(rule, houses, weights, 0.6, grid=401)
        same, _ = M.target_equilibrium(phantoms, houses, weights, 0.6, grid=401)
        assert np.allclose(targets, same)
        assert np.isclose(M.social_loss(lambda x: M.step(rule, x, targets, weights, 0.6), houses),
                          M.social_loss(lambda x: M.step(phantoms, x, same, weights, 0.6), houses))
    targets, _ = M.target_equilibrium((0.1, 0.3, 0.7, 0.9), houses, weights, 0.6, grid=401)
    policy = lambda v: M.reports((0.1, 0.3, 0.7, 0.9), v, targets, weights, 0.6)  # noqa: E731
    assert M.deviation_gain((0.1, 0.3, 0.7, 0.9), policy, houses, weights, 0.6, **SMALL) < 1e-5


def test_without_inertia_the_three_player_loss_of_phantoms_p_and_1_minus_p_matches_the_closed_form():
    houses = np.random.default_rng(0).random((200_000, 3))
    for p in (0.0, 0.2, 1 / 3, 0.5):
        both = np.concatenate([houses, np.full((len(houses), 1), p), np.full((len(houses), 1), 1 - p)], 1)
        z = np.median(both, axis=1)
        exact = (1 - 10 * p ** 3 + 30 * p ** 4 - 18 * p ** 5) / 5
        assert abs(((z[:, None] - houses) ** 2).sum(1).mean() - exact) < 2e-3


def test_the_high_inertia_coefficients_match_simulation_and_pick_the_median_receiver():
    rng = np.random.default_rng(1)
    houses, x = rng.random((400_000, 5)), rng.random(400_000)
    above, gap = (houses > x[:, None]).sum(1), x - houses.mean(1)
    assert np.allclose([np.mean(gap * (above == k)) for k in range(6)], M.limit_coefficients(5), atol=2e-3)
    for n in (3, 5, 7):
        c = M.limit_coefficients(n)[1:n]  # the free phantoms alpha_1 .. alpha_{n-1}
        thresholds = [sum(c[m:]) for m in range(n)]  # alpha_k = 1 from k = m + 1 on: the extreme points
        assert int(np.argmin(thresholds)) == (n - 1) // 2  # the first (n - 1) / 2 phantoms at 0, the rest at 1


def test_the_best_slow_receiver_is_the_order_statistic_closest_to_the_mean_target():
    rng = np.random.default_rng(3)
    houses, x = rng.beta(0.2, 5, (300_000, 5)), rng.random(300_000)
    ordered, mean, above = np.sort(houses, 1), houses.mean(1), (houses > x[:, None]).sum(1)
    for m in range(1, 6):  # the tail sum of the c_k equals E[q (q / 2 - mean)] with q the (6 - m)-th smallest target
        q = ordered[:, 5 - m]
        assert abs(np.mean((x - mean) * (above >= m)) - np.mean(q * (q / 2 - mean))) < 2e-3
    assert M.best_order_statistic(rng.random((300_000, 5))) == 3  # uniform targets: the median
    assert M.best_order_statistic(houses) == 4  # skewed targets: a higher order statistic
    assert M.order_statistic_phantoms(5, 3) == (0.0, 0.0, 1.0, 1.0) and M.order_statistic_phantoms(5, 4) == (0.0, 1.0, 1.0, 1.0)
