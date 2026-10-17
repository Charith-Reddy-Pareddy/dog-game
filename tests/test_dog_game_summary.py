from doggame.dog_game_summary import (walking_entropy_table, pretraining_mechanism_table, noisy_inertia_table, exact_play_table, feedback_learning_table, many_houses_table, inertia_dp_table, inertia_explicit_table, inertia_random_table, mirror_p, selection_clip_table, markov_potential_table, selection_noise_table, algorithms_table, convergence_counts, markdown, noise_table, resolution_table,
                                      selection_outcome, selection_table, walking_algorithms_table, walking_share_series, walking_table,
                                      ppo_epochs_table, ppo_long_table, ppo_walking_table, walking_equilibrium_table, walking_houses_table, walking_inertia_table,
                                      walking_settings_table)


def run(study, key, **fields):
    return dict(study=study, key=key, **fields)


def test_selection_outcome_names_the_equilibrium_a_policy_is_at():
    assert selection_outcome([1.0, 1.0], [0.0, 0.0]) == "red top, blue bottom"
    assert selection_outcome([1.0, 0.0], [0.0, 1.0]) == "red bottom, blue top"
    assert selection_outcome([1.0, 0.5], [0.0, 0.5]) == "both at y = 0.5"
    assert selection_outcome([1.0, 0.7], [0.0, 0.3]) == "other"


def test_the_selection_table_counts_each_outcome_per_layout():
    rows = [run("selection", dict(architecture="separate", seed=s), red=r, blue=b, exploitability=0.0001)
            for s, (r, b) in enumerate([([1, 1], [0, 0]), ([1, 1], [0, 0]), ([1, 0], [0, 1])])]

    table = selection_table(rows)

    assert table[1][:5] == ["separate", "2 of 3", "1 of 3", "0 of 3", "0 of 3"]


def test_the_algorithms_table_shares_converged_runs_at_each_checkpoint():
    rows = [run("algorithms", dict(config=0, algorithm="ppo", seed=s), curve={"100": e100, "500": 0.0001})
            for s, e100 in enumerate([0.001, 0.5])]

    table = algorithms_table(rows)

    assert table[0][:3] == ["Algorithm", "100 episodes", "500 episodes"]
    assert table[1][:3] == ["PPO", "1 of 2", "2 of 2"]


def test_resolution_table_counts_unique_setups_and_reports_the_shared_y_setup_separately():
    def entry(config, resolution, distance):
        return run("resolution", dict(config=config, resolution=resolution), actions=resolution ** 2, support=[1, 1], distance=distance, seconds=0.1)

    rows = [entry(0, 3, 0.0), entry(1, 3, 1.5), entry(2, 3, 0.0), entry(3, 3, 1.0), entry(4, 3, 0.0)]

    row = resolution_table(rows)[1]

    assert row[0] == "3 x 3" and row[3] == "3" and row[4] == "1.00" and row[5] == "yes"


def test_noise_table_reports_the_leftover_weight_of_the_first_pick():
    rows = [run("noise", dict(iterations=30, seed=0), spread=0.64, gap=0.0, support=2, top=30 / 31),
            run("noise", dict(iterations="exact", seed=0), spread=0.0, gap=0.0, support=1, top=1.0)]

    table = noise_table(rows)

    assert table[1] == ["30", "0.6400", "2", "0.03226"] and table[2][0] == "exact (Lemke-Howson)"


def test_convergence_counts_and_markdown():
    runs = [dict(kind="convergence", architecture="shared", exploitability=0.001),
            dict(kind="convergence", architecture="shared", exploitability=0.3), dict(kind="walking")]

    assert convergence_counts(runs) == ((1, 2), {"shared": (1, 2)})
    assert markdown([["a", "b"], ["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_walking_table_averages_each_method_over_its_seeds():
    runs = [dict(kind="walking", architecture="reinforce", before=1.0, after=0.1), dict(kind="walking", architecture="reinforce", before=2.0, after=0.3),
            dict(kind="walking", architecture="nash-q", before=1.0, after=1.5), dict(kind="convergence")]

    table = walking_table(runs)

    assert table[1] == ["REINFORCE (angle and step)", "1.50", "0.20", "2 of 2"]
    assert table[2] == ["Nash-Q (10 directions)", "1.00", "1.50", "0 of 1"]


def test_walking_algorithms_table_averages_distances_and_counts_time_at_the_corners():
    curves = [{"100": 0.5, "150": 0.02, "300": 0.03}, {"100": 0.3, "150": 0.5, "300": 0.2}]
    rows = [run("walking", dict(algorithm="ppo", seed=s), start=1.5, curve=curve) for s, curve in enumerate(curves)]

    table = walking_algorithms_table(rows)

    assert table[0][:4] == ["Algorithm", "Before training", "At 100", "At 300"]
    assert table[1] == ["PPO", "1.50", "0.400", "0.115", "2 of 4", "1 of 2"]  # 150 and 300 for each seed: 0.02 and 0.03 are under 0.1


def test_the_share_series_shows_a_seed_leaving_the_corners_again():
    rows = [run("walking", dict(algorithm="ppo", seed=0), start=1.0, curve={"50": 0.01, "100": 0.02}),
            run("walking", dict(algorithm="ppo", seed=1), start=1.0, curve={"50": 0.02, "100": 0.9})]

    assert walking_share_series(rows) == {"ppo": [(50, 1.0), (100, 0.5)]}


def test_ppo_epochs_table_lines_up_gradient_steps_with_the_share_of_converged_runs():
    def entry(study_name, key, e100):
        return run(study_name, key, curve={"100": e100, "500": 0.0001})

    rows = [entry("algorithms", dict(config=0, algorithm="reinforce", seed=0), 0.5), entry("algorithms", dict(config=0, algorithm="ppo", seed=0), 0.001),
            entry("ppo_variants", dict(game="main", epochs=1, lr=0.01, config=0, seed=0), 0.001),
            entry("ppo_variants", dict(game="walking", epochs=1, lr=0.01, seed=0), 0.5)]  # a walking run is not a main-game run

    table = ppo_epochs_table(rows)

    assert [r[0] for r in table[1:]] == ["REINFORCE", "PPO, 1 epoch", "PPO, 4 epochs (default)"]
    assert table[1][:4] == ["REINFORCE", "1", "0 of 1", "1 of 1"] and table[2][:3] == ["PPO, 1 epoch", "1", "1 of 1"] and table[3][1] == "4"


def test_ppo_walking_table_compares_settings_with_the_default_and_a2c():
    curve = {"100": 0.05, "150": 0.02, "300": 0.03}
    rows = [run("walking", dict(algorithm="a2c", seed=0), start=1.5, curve=curve), run("walking", dict(algorithm="ppo", seed=0), start=1.5, curve=curve),
            run("ppo_variants", dict(game="walking", epochs=4, lr=0.003, seed=0), start=1.5, curve=curve)]

    table = ppo_walking_table(rows)

    assert [r[0] for r in table[1:]] == ["A2C (reference)", "PPO, lr 0.01, 4 epochs (default)", "PPO, lr 0.003, 4 epochs"]
    assert table[3][-2:] == ["2 of 2", "1 of 1"]


def test_ppo_long_table_uses_the_longer_checkpoints_and_a_later_start_for_the_late_share():
    curve = {"300": 0.5, "450": 0.02, "600": 0.4, "900": 0.03}
    rows = [run("ppo_long", dict(setting="ppo4", seed=0), start=1.5, curve=curve), run("ppo_long", dict(setting="a2c", seed=0), start=1.5, curve=curve)]

    table = ppo_long_table(rows)

    assert table[0][:5] == ["Setting", "Before training", "At 300", "At 600", "At 900"]
    assert [r[0] for r in table[1:]] == ["A2C", "PPO, 4 epochs (default)"]
    assert table[1][-2:] == ["2 of 3", "1 of 1"]  # checkpoints 450, 600 and 900: two are under 0.1


def test_walking_settings_table_puts_the_default_next_to_the_changed_settings():
    curve = {"50": 0.9, "100": 0.5, "200": 0.1, "300": 0.05}
    rows = [run("walking", dict(algorithm="a2c", seed=0), start=1.5, curve=curve),
            run("walking_settings", dict(max_step=0.4, discount=0.9, seed=0), start=1.5, curve=curve)]

    table = walking_settings_table(rows)

    assert [r[0] for r in table[1:]] == ["max step 0.2, discount 0.9 (default)", "max step 0.4, discount 0.9"]


def test_walking_houses_table_lists_the_default_setup_and_then_each_other_setup_per_algorithm():
    curve = {"100": 0.4, "150": 0.02, "300": 0.03}
    rows = [run("walking", dict(algorithm="a2c", seed=0), start=1.5, curve=curve),
            run("walking_houses", dict(setup=1, algorithm="a2c", seed=0), start=1.5, curve=curve),
            run("walking_houses", dict(setup=1, algorithm="ppo", seed=0), start=1.5, curve=curve)]

    table = walking_houses_table(rows)

    assert table[0][:4] == ["House setup", "Algorithm", "Before training", "At 100"]
    assert [(r[0].startswith("default"), r[1]) for r in table[1:]] == [(True, "A2C"), (False, "A2C"), (False, "PPO")]
    assert "(0.8, 0.8) / (0.2, 0.3)" in table[2][0] and table[2][-2:] == ["2 of 2", "1 of 1"]


def test_walking_inertia_table_puts_the_default_next_to_the_inertia_settings():
    curve = {"50": 0.9, "100": 0.5, "200": 0.1, "300": 0.05}
    rows = [run("walking", dict(algorithm="a2c", seed=0), start=1.5, curve=curve), run("walking_inertia", dict(w0=0.6, seed=0), start=1.5, curve=curve)]

    assert [r[0] for r in walking_inertia_table(rows)[1:]] == ["w0 = 0 (default)", "w0 = 0.6"]


def test_walking_equilibrium_table_shows_each_players_gain_and_the_largest_share_of_the_reward():
    result = lambda gain: {"followed": -2.0, "best": -2.0 + gain, "gain": gain}
    rows = [run("walking_equilibrium", dict(setup="default", start="houses"), red=result(0.0), blue=result(0.02)),
            run("walking_equilibrium", dict(setup=2, start="corners"), red=result(0.0), blue=result(0.0))]

    table = walking_equilibrium_table(rows)

    assert table[1][1:] == ["on their houses", "0.00000", "0.02000", "1.00%"]
    assert table[2][0].startswith("(0.7, 0.1)") and table[2][1:] == ["on the corners", "0.00000", "0.00000", "0.00%"]


def test_the_selection_noise_table_puts_the_default_next_to_the_other_noise_levels():
    row = lambda study_name, key, red, blue: run(study_name, key, red=red, blue=blue, exploitability=0.0001)
    rows = [row("selection", dict(architecture="separate", seed=0), [1, 1], [0, 0]),
            row("selection_noise", dict(log_std=-2.0, seed=0), [1, 0], [0, 1])]

    table = selection_noise_table(rows)

    assert [r[0] for r in table[1:]] == ["0.14", "0.61 (default)"]
    assert table[1][1:3] == ["0 of 1", "1 of 1"] and table[2][1:3] == ["1 of 1", "0 of 1"]


def test_the_dp_and_cycle_tables_show_their_measurements():
    rows = [run("inertia_dp", dict(setup=0, w0=0.6), path_gap=0.0, differs=[0.0, 0.25], unresolved=[0, 0], sweeps=[10, 10]),
            run("markov_potential", dict(w0=0.6, policies="fixed"), cycle=1e-15),
            run("markov_potential", dict(w0=0.6, policies="reacting"), cycle=5.625)]

    assert inertia_dp_table(rows)[1][1] == "0.00 / 0%, 25%"
    assert markov_potential_table(rows)[1] == ["0.6", "+1.0e-15", "+5.62"]


def test_the_mirror_test_is_even_for_a_balanced_split_and_small_for_a_lopsided_one():
    assert mirror_p(25, 25) == 1.0 and mirror_p(45, 5) < 1e-6


def test_the_clip_table_compares_clipped_and_unclipped_runs():
    rows = [run("selection_noise", dict(log_std=-0.5, seed=0), red=[1, 1], blue=[0, 0]),
            run("selection_clip", dict(seed=0), red=[1, 0.5], blue=[0, 0.5]), run("selection_clip", dict(seed=1), red=[1, 0.4], blue=[0, 0.6]),
            run("selection_linear", dict(seed=0), red=[1, 3.0], blue=[0, -2.0])]
    table = selection_clip_table(rows)
    assert [r[0] for r in table[1:]] == ["sigmoid, clipped (default)", "sigmoid, not clipped", "linear, not clipped"]
    assert table[2][3] == "2 of 2" and table[2][-2:] == ["0.43 / 0.45 / 0.47", "0 of 2"] and table[3][-1] == "1 of 1"


def test_the_explicit_equilibrium_tables_report_shares_gaps_and_gains():
    rows = [run("inertia_explicit", dict(setup=3, w0=0.6), kind=["straddle", "shared"], theory=[0.0, 1.0], dp=[0.0, 1.0], gap=[0.0, 0.5], gain=[0.0, 0.01]),
            run("inertia_random", dict(game=0), kind="straddle", gain=0.0, holds=True), run("inertia_random", dict(game=1), kind="interior", gain=0.003, holds=False)]
    assert inertia_explicit_table(rows)[1][1] == "x 0% / 0%; 0.00, y 100% / 100%; shared"
    assert inertia_random_table(rows)[1:] == [["straddle", "1", "1 of 1", "0.0e+00", "1 of 1"], ["interior", "1", "0 of 1", "3.0e-03", "0 of 1"]]


def test_the_many_houses_table_reports_the_dog_pivots_and_critical_inertia():
    rows = [run("many_houses", dict(n=4), dog=[0.3, 0.4], pivot=[True, False], critical=[0.0, 0.1])]
    table = many_houses_table(rows)
    assert table[0][1].endswith("limit 0.339") and table[1] == ["4", "0.350 (0.050)", "0.50", "0.0500", "0.40"]


def test_the_feedback_learning_table_puts_baselines_before_learners():
    rows = [run("feedback_learning", dict(algorithm="holding", seed=0), max_gain=0.0756, mean_gain=0.0114),
            run("feedback_learning", dict(algorithm="a2c", seed=0), max_gain=0.07, mean_gain=0.01),
            run("feedback_learning", dict(algorithm="a2c", seed=1), max_gain=0.05, mean_gain=0.02)]
    table = feedback_learning_table(rows)
    assert table[1] == ["setup 3, w0 = 0.6", "hold the one-shot actions", "1", "0.0756", "0.0114"]
    assert table[2] == ["setup 3, w0 = 0.6", "A2C", "2", "0.0600 (0.0500 to 0.0700)", "0.0150"]


def test_the_exact_play_table_lists_tables_before_networks():
    key = lambda kind, squash, hidden, seed: dict(setup=2, w0=0.6, kind=kind, squash=squash, hidden=hidden, seed=seed)
    rows = [run("exact_play", key("network", "sigmoid", 16, 0), gap_explicit=1.0, gap_holding=0.0, gain=0.0756),
            run("exact_play", key("network", "sigmoid", 16, 1), gap_explicit=0.6, gap_holding=1.0, gain=0.03),
            run("exact_play", key("table", "sigmoid", 16, 0), gap_explicit=0.015, gap_holding=1.0, gain=0.0007)]
    table = exact_play_table(rows)
    assert table[1] == ["setup 3 x, w0 = 0.6", "table, sigmoid", "1", "1 of 1", "0 of 1", "0.0007"]
    assert table[2] == ["setup 3 x, w0 = 0.6", "network, 16 units", "2", "0 of 2", "1 of 2", "0.0528 (0.0300 to 0.0756)"]


def test_the_noisy_inertia_table_puts_the_closed_form_and_limit_first():
    rows = [run("noisy_inertia", dict(dt=0.05, sigma=0.2, clipped=True), red=0.765, blue=0.545, sweeps=559, unresolved=0),
            run("noisy_inertia", dict(dt=0.05, sigma=0.4, clipped=True), red=0.79, blue=0.59, sweeps=8000, unresolved=0)]
    table = noisy_inertia_table(rows)
    assert table[1][:3] == ["closed form (Proposition 2)", "0.05", "0"] and table[2] == ["continuous-time limit", "0", "0", "0.700", "0.533", "-"]
    assert table[3][3:] == ["0.765", "0.545", "559 sweeps"] and table[4][-1] == "did not converge"


def test_the_pretraining_mechanism_table_counts_runs_under_the_threshold():
    rows = [run("pretraining_mechanism", dict(variant="scratch", seed=0), curve={"100": 0.05, "300": 0.003}, saturation=None),
            run("pretraining_mechanism", dict(variant="pretrained", seed=0), curve={"100": 0.3, "300": 0.2}, saturation=dict(raw=8.5, log_std=-2.8))]
    table = pretraining_mechanism_table(rows)
    assert table[1] == ["scratch", "0 of 1", "1 of 1", "0.0030", "-"] and table[2][-1] == "8.5, -2.80"


def test_the_walking_entropy_table_has_a_row_per_coefficient():
    curve = {"300": 0.05, "450": 0.04, "600": 0.03, "900": 0.02}
    rows = [run("walking_entropy", dict(entropy=c, seed=0), start=1.0, curve=curve) for c in (0.0, 0.1)]
    table = walking_entropy_table(rows)
    assert [r[0] for r in table[1:]] == ["0.0", "0.1"] and table[1][-1] == "1 of 1"
