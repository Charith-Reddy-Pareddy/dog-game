import json

import numpy as np

import doggame.dog_game_studies as studies
from doggame.agents import TwoPlayerPolicy
from doggame.exact_nash import lemke_howson_nash


def test_the_gap_is_zero_at_the_exact_equilibrium_and_positive_away_from_it():
    _, payoff_red, payoff_blue = studies.stage_game(5)
    strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)
    uniform = np.full(len(strategy_red), 1 / len(strategy_red))

    assert studies.nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue) < 1e-9
    assert studies.nash_gap(payoff_red, payoff_blue, uniform, uniform) > 0.01


def test_a_score_reports_the_distance_and_exploitability_of_the_policy():
    env = studies.make_env(studies.DEFAULT_CONFIGS[0])

    result = studies.score(TwoPlayerPolicy(env.domain, seed=0), env)

    assert set(result) == {"exploitability", "distance", "red", "blue"}
    assert result["exploitability"] > 0 and len(result["red"]) == 2


def test_fictitious_play_looks_mixed_only_while_its_first_random_pick_is_above_the_floor():
    rows = {(key["iterations"], key["seed"]): execute() for key, execute in studies.study_noise([0])}

    assert rows[("exact", 0)]["spread"] < 1e-9 and rows[("exact", 0)]["support"] == 1
    assert rows[(30, 0)]["spread"] > 0.1 and rows[(30, 0)]["support"] == 2   # the leftover 1/31 of the first pick
    assert rows[(30, 0)]["top"] == np.float64(30 / 31) or abs(rows[(30, 0)]["top"] - 30 / 31) < 1e-9
    assert rows[(10000, 0)]["spread"] < 1e-9 and rows[(10000, 0)]["support"] == 1


def test_the_resolution_study_finds_the_continuous_equilibrium_on_every_grid():
    rows = [execute() for key, execute in studies.study_resolution([0]) if key["config"] == 0]

    assert [r["actions"] for r in rows] == [9, 25, 81, 169, 289]
    assert all(r["support"] == [1, 1] and r["distance"] < 1e-6 for r in rows)


def test_run_skips_finished_runs_and_appends_the_rest(tmp_path, monkeypatch):
    calls = []

    def tiny_study(seeds):
        for seed in seeds:
            yield dict(seed=seed), lambda seed=seed: (calls.append(seed), dict(value=seed * 10))[1]

    monkeypatch.setitem(studies.STUDIES, "tiny", tiny_study)
    path = tmp_path / "studies.jsonl"
    path.write_text(json.dumps(dict(study="tiny", key=dict(seed=0), value=0)) + "\n")

    studies.run(["tiny"], seeds=range(3), path=str(path))

    assert calls == [1, 2]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["key"]["seed"] for r in rows] == [0, 1, 2] and rows[2]["value"] == 20


def test_shards_split_the_runs_between_processes_without_overlap(tmp_path, monkeypatch):
    def tiny_study(seeds):
        for seed in seeds:
            yield dict(seed=seed), lambda seed=seed: dict(value=seed)

    monkeypatch.setitem(studies.STUDIES, "tiny", tiny_study)
    path = str(tmp_path / "studies.jsonl")

    studies.run(["tiny"], seeds=range(6), path=path, shard=(1, 3))
    studies.run(["tiny"], seeds=range(6), path=path, shard=(0, 3))

    seeds = [json.loads(line)["key"]["seed"] for line in open(path)]
    assert seeds == [1, 4, 0, 3]  # shard 1 of 3 did runs 1 and 4, shard 0 did runs 0 and 3, nobody did 2 or 5


def test_the_ppo_variants_study_covers_both_games_with_distinct_settings():
    keys = [key for key, _ in studies.study_ppo_variants(range(5))]

    main = [k for k in keys if k["game"] == "main"]
    walking = [k for k in keys if k["game"] == "walking"]
    assert len(main) == 2 * 5 * 5 and {k["epochs"] for k in main} == {1, 2}   # 1 and 2 epochs, 5 setups, 5 seeds
    assert sorted({(k["lr"], k["epochs"]) for k in walking}) == [(0.003, 1), (0.003, 4), (0.01, 1)] and len(walking) == 15
    assert len({json.dumps(k, sort_keys=True) for k in keys}) == len(keys)  # no run is listed twice


def test_the_walking_studies_define_the_expected_runs_without_overlap():
    long_keys = [k for k, _ in studies.study_ppo_long(range(5))]
    settings = [k for k, _ in studies.study_walking_settings(range(5))]

    assert sorted({k["setting"] for k in long_keys}) == ["a2c", "ppo1", "ppo4"] and len(long_keys) == 15
    assert sorted({(k["max_step"], k["discount"]) for k in settings}) == [(0.1, 0.9), (0.2, 0.5), (0.2, 0.99), (0.4, 0.9)] and len(settings) == 20


def test_the_walking_house_and_inertia_studies_define_the_expected_runs():
    houses = [k for k, _ in studies.study_walking_houses(range(5))]
    inertia = [k for k, _ in studies.study_walking_inertia(range(5))]

    assert len(houses) == 3 * 3 * 5 and {k["setup"] for k in houses} == {0, 1, 2} and {k["algorithm"] for k in houses} == {"reinforce", "a2c", "ppo"}
    assert len(inertia) == 10 and {k["w0"] for k in inertia} == {0.3, 0.6}


def test_the_walking_equilibrium_study_checks_four_setups_resting_and_walking():
    keys = [k for k, _ in studies.study_walking_equilibrium(range(5))]

    assert len(keys) == 8 and {k["start"] for k in keys} == {"houses", "corners"} and {k["setup"] for k in keys} == {"default", 0, 1, 2}


def test_the_selection_noise_study_varies_only_the_initial_noise():
    keys = [k for k, _ in studies.study_selection_noise(range(5))]

    assert len(keys) == 200 and sorted({k["log_std"] for k in keys}) == [-2.0, -1.0, -0.5, 0.0]


def test_the_clip_ablation_leaves_actions_and_the_dog_unclipped():
    assert len([k for k, _ in studies.study_selection_clip(range(5))]) == 50
    assert np.allclose(studies.Unclipped().clip([1.4, -0.2]), [1.4, -0.2])


def test_the_explicit_equilibrium_studies_cover_every_setup_and_200_random_games():
    assert len([k for k, _ in studies.study_inertia_explicit(range(5))]) == 15
    assert len([k for k, _ in studies.study_inertia_random(range(5))]) == 200


def test_the_dp_and_markov_potential_studies_define_their_checks():
    dp = [k for k, _ in studies.study_inertia_dp(range(5))]
    cycles = [k for k, _ in studies.study_markov_potential(range(5))]

    assert len(dp) == 15 and {k["w0"] for k in dp} == {0.3, 0.6, 0.9}
    assert len(cycles) == 8 and {k["policies"] for k in cycles} == {"fixed", "reacting"}


def test_the_many_houses_study_covers_two_to_1024_houses():
    assert [k["n"] for k, _ in studies.study_many_houses(range(5))] == [2 ** k for k in range(1, 11)]


def test_the_feedback_learning_study_has_two_baselines_and_three_algorithms():
    keys = [k for k, _ in studies.study_feedback_learning(range(5))]
    assert [k["algorithm"] for k in keys[:2]] == ["explicit", "holding"] and len(keys) == 2 + 15 + 15 + 6 + 2 + 15
    assert {k.get("family", "gaussian") for k in keys[2:]} == {"gaussian", "beta"} and max(k.get("episodes", 3000) for k in keys) == 15000


def test_the_exact_play_study_compares_tables_with_networks_in_two_settings():
    keys = [k for k, _ in studies.study_exact_play(range(5))]
    assert len(keys) == 2 * (2 + 10) and {k["kind"] for k in keys} == {"table", "network"}


def test_the_noisy_inertia_study_varies_the_noise_the_time_step_and_the_clipping():
    keys = [k for k, _ in studies.study_noisy_inertia(range(5))]
    assert len(keys) == 12 and {k["clipped"] for k in keys} == {True, False} and {k["dt"] for k in keys} == {0.05, 0.025}
