from doggame.dog_game_summary import (algorithms_table, convergence_counts, markdown, noise_table, walking_table, resolution_table,
                                      selection_outcome, selection_table)


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
