from doggame.dog_game_summary import (algorithms_table, convergence_counts, markdown, noise_table, resolution_table,
                                      selection_outcome, selection_table, walking_algorithms_table, walking_share_series, walking_table,
                                      ppo_epochs_table, ppo_long_table, ppo_walking_table, walking_settings_table)


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
