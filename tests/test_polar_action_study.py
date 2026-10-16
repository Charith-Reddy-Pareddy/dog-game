import numpy as np

from doggame.agents import BoundedCartesianPolicy, PolarActionPolicy, TwoPlayerPolicy
from doggame.env import DogGameEnv
from doggame.experiments import DEFAULT_CONFIGS
from doggame.nash import solve_stage_nash
from doggame.polar_action_study import action_moments, expected_profile_gap
from doggame.polar_action_robustness import CONFIGS, REPRESENTATIONS


def test_moment_based_profile_gap_is_zero_at_the_analytic_pure_equilibrium():
    config = DEFAULT_CONFIGS[0]
    env = DogGameEnv(**config)
    red, blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    moments = dict(red_mean=red, blue_mean=blue, red_var=0.0, blue_var=0.0)
    assert max(expected_profile_gap(moments, env)) < 1e-12


def test_moment_based_profile_gap_counts_own_policy_variance_as_deviation_gain():
    config = DEFAULT_CONFIGS[0]
    env = DogGameEnv(**config)
    red, blue, _ = solve_stage_nash(env.house_red, env.house_blue, env.w, env.domain)
    moments = dict(red_mean=red, blue_mean=blue, red_var=0.01, blue_var=0.0)
    gaps = expected_profile_gap(moments, env)
    assert np.isclose(gaps[0], env.w**2 * 0.01)
    assert gaps[1] == 0.0


def test_action_moment_scorer_accepts_a_single_unbatched_state_for_each_policy():
    env = DogGameEnv(**DEFAULT_CONFIGS[0])
    state = env.reset()
    policies = (
        (TwoPlayerPolicy(env.domain, seed=0), "cartesian_gaussian_clipped"),
        (BoundedCartesianPolicy(env.domain, seed=0), "cartesian_beta"),
        (PolarActionPolicy(env.domain, seed=0), "polar_beta"),
    )
    for policy, representation in policies:
        moments = action_moments(policy, state, representation, samples=32, seed=11)
        assert moments["red_mean"].shape == moments["blue_mean"].shape == (2,)
        assert np.isfinite(moments["red_var"]) and np.isfinite(moments["blue_var"])


def test_cross_house_followup_uses_all_nonprimary_configs_and_bounded_controls():
    assert CONFIGS == (1, 2, 3, 4)
    assert set(REPRESENTATIONS) == {"cartesian_beta", "polar_beta"}
