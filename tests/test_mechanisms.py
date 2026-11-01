from math import comb

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


def test_truthful_reporting_is_an_equilibrium_once_the_receiver_compensates_for_inertia():
    truthful = lambda x: HOUSES.copy()
    for phantoms in ((0.0, 1.0), (1 / 3, 2 / 3), (0.2, 0.9)):
        for w0 in (0.3, 0.6, 0.9):
            assert M.deviation_gain(phantoms, truthful, HOUSES, WEIGHTS, w0, compensate=True, **SMALL) < 1e-12
        assert M.deviation_gain(phantoms, truthful, HOUSES, WEIGHTS, 0.6, **SMALL) > 0.01


def test_compensation_keeps_every_strategy_proof_rule_truthful_but_not_averaging():
    truthful = lambda x: HOUSES.copy()
    dictator = lambda r: r[..., 0]
    lattice = lambda r: np.maximum(np.minimum(r[..., 0], 0.7), np.minimum(r[..., 1], r[..., 2]))  # not anonymous
    for rule in (dictator, lattice):
        assert M.deviation_gain(rule, truthful, HOUSES, WEIGHTS, 0.6, compensate=True, **SMALL) < 1e-12
    assert M.deviation_gain(lambda r: r.mean(-1), truthful, HOUSES, WEIGHTS, 0.6, compensate=True, **SMALL) > 0.01


def test_compensation_keeps_truthful_reporting_for_any_monotone_reachable_set_that_allows_staying():
    truthful = lambda x: HOUSES.copy()
    speed_limit = lambda x: (np.maximum(0, x - 0.15), np.minimum(1, x + 0.15))
    asymmetric = lambda x: (np.maximum(0, x - 0.3), np.minimum(1, x + 0.1))  # falls fast, rises slowly
    for reach in (speed_limit, asymmetric):
        for rule in ((0.0, 1.0), (1 / 3, 2 / 3), lambda r: r[..., 0]):
            assert M.deviation_gain(rule, truthful, HOUSES, WEIGHTS, 0.0, compensate=reach, **SMALL) < 1e-12


def test_truthful_reporting_fails_when_the_receiver_cannot_stay_or_its_reach_falls():
    dictator, one = (lambda r: r[..., 0]), np.ones(1)
    forced = lambda x: (np.minimum(1, x + 0.2), np.minimum(1, x + 0.4))  # must move up by at least 0.2
    falling = lambda x: (x, np.where(x <= 0.45, np.minimum(1, x + 0.6), x))  # the upper end falls from 1 to 0.45 above 0.45
    gain = lambda reach, target: M.deviation_gain(dictator, lambda x: np.full(1, target), np.full(1, target), one, 0.0,
                                                  states=11, grid=11, rounds=200, compensate=reach)
    # from start 0 (one of the states checked), by hand: report 0.3 first and move 0.3, 0.5, 0.7, ... instead of
    # 0.4, 0.6, 0.8, ... (0.085 less loss); or report 0.4 first and then reach 1 for good instead of being stuck at 0.6
    assert gain(forced, 0.5) >= 0.085 and gain(falling, 1.0) >= 1.24 - 1e-6  # 200 rounds leave out about 1e-9


def test_a_slow_compensated_receiver_is_ranked_by_the_static_criterion():
    rng = np.random.default_rng(2)
    houses, x = rng.random((400_000, 3)), rng.random(400_000)
    s = np.median(np.concatenate([houses, np.full((len(x), 1), 0.2), np.full((len(x), 1), 0.8)], 1), 1)
    assert abs(np.mean((x - houses.mean(1)) * (s > x)) - M.compensated_criterion(houses, (0.2, 0.8))) < 2e-3
    static = lambda p: (1 - 10 * p ** 3 + 30 * p ** 4 - 18 * p ** 5) / 5  # E sum_i (s - h_i)^2 = 3 E (s - hbar)^2 + const
    draws = rng.random((300, 3))
    loss = lambda p, w0: np.mean([M.social_loss(lambda v: M.compensated(v, np.tile(h, (len(v), 1)), (p, 1 - p), w0), h) for h in draws])
    gap = loss(0.0, 0.995) - loss(1 / 3, 0.995)
    assert abs(gap / (0.005 / 0.01 * (static(0.0) - static(1 / 3))) - 1) < 0.15


def test_the_batched_compensated_losses_match_the_social_loss():
    shapes = [(0.0, 1.0), (1 / 3, 2 / 3), (0.2, 0.9)]
    for w0 in (0.0, 0.6, 0.95):
        each = [M.social_loss(lambda x, ph=ph: M.compensated(x, np.tile(HOUSES, (len(x), 1)), ph, w0), HOUSES) for ph in shapes]
        assert np.allclose(M.compensated_losses(HOUSES, shapes, w0), each)


def test_a_compensated_receiver_loses_the_planners_loss_plus_a_weighted_static_criterion():
    mean = HOUSES.mean()
    for w0 in (0.0, 0.6, 0.95):
        ends = np.array([0.1, mean, 0.8])
        losses = M.compensated_losses(HOUSES, [(e, e) for e in ends], w0)  # two phantoms at e: the end point is e
        assert np.isclose(losses[1], M.social_loss(lambda x: M.planner_step(x, HOUSES, w0), HOUSES))
        for e, loss in zip(ends[[0, 2]], losses[[0, 2]]):
            u = np.linspace(mean, e, 20001)
            integral = np.trapz(6 * (u - mean) * M.reach_weight(u, w0), u)  # W is a step function: quadrature error ~ 1e-6
            assert np.isclose(loss - losses[1], integral, rtol=1e-4)
    assert np.allclose(M.reach_weight(np.linspace(0, 1, 5), 0.0), (1 - 0.9 ** 200) / 0.1)  # no inertia: always reachable


def test_moving_a_phantom_changes_the_loss_at_the_static_rate_times_the_weight():
    rng = np.random.default_rng(3)
    houses, alpha, step = rng.random((200, 3)), np.array([0.22, 0.71]), 1e-4
    loss = lambda a, w0: np.mean([M.compensated_losses(h, [tuple(a)], w0)[0] for h in houses])
    s = np.median(np.concatenate([houses, np.tile(alpha, (len(houses), 1))], 1), 1)
    for w0 in (0.0, 0.6, 0.9):
        for k in range(2):
            shift = step * np.eye(2)[k]
            numeric = (loss(alpha + shift, w0) - loss(alpha - shift, w0)) / (2 * step)
            rate = 6 * M.reach_weight(alpha[k], w0) * np.mean((alpha[k] - houses.mean(1)) * np.isclose(s, alpha[k]))
            assert abs(numeric - rate) < 1e-3 * abs(rate)


def test_for_uniform_targets_the_phantom_condition_has_its_closed_form_and_vanishes_at_k_over_n():
    rng = np.random.default_rng(4)
    for n in (3, 5):
        houses = rng.random((2_000_000, n))
        for k in range(1, n):
            for a in (k / n, k / n + 0.1):
                below = (houses < a).sum(1) == n - k  # the phantom is the median exactly when n - k targets lie below it
                sample = np.mean((a - houses.mean(1)) * below)
                closed = comb(n, k) * a ** (n - k) * (1 - a) ** k * (a - k / n) / 2
                assert abs(sample - closed) < 4e-4


def test_with_clipped_shocks_the_median_influencer_gains_a_little_by_shading():
    houses = np.array([0.15, 0.3, 0.8])
    reports = np.tile(houses, (201, 1))
    reports[:, 1] = np.linspace(0, 1, 201)  # the median influencer's reports; index 60 is its target 0.3

    def gain(x, w0):
        move = lambda v: M.compensated(v, np.broadcast_to(houses, (len(v), 3)), (0.0, 1.0), w0)
        z = M.compensated(np.full(201, x), reports, (0.0, 1.0), w0)
        losses = M._shock_losses(move, z, houses, 0.9, M.shocks(0.15, w0, filtered=False))[1]
        return losses[60] - losses.min()

    assert abs(gain(0.125, 0.8) - 0.0024) < 1e-4 and abs(gain(0.0, 0.0) - 0.0003) < 5e-5


def test_the_compensated_median_with_truthful_reports_follows_the_median_rule_equilibrium():
    x = np.linspace(0, 1, 11)
    path = M.compensated(x, np.tile(HOUSES, (len(x), 1)), (0.0, 1.0), 0.6)
    assert np.allclose(path, np.clip(0.45, 0.6 * x, 0.6 * x + 0.4))


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
