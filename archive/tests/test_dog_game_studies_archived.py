"""Archived tests: frozen copies removed from tests/test_dog_game_studies.py. They are not imported or tested here; run them at the
git tag pre-archive."""


def test_the_gap_is_zero_at_the_exact_equilibrium_and_positive_away_from_it():
    _, payoff_red, payoff_blue = studies.stage_game(5)
    strategy_red, strategy_blue = lemke_howson_nash(payoff_red, payoff_blue)
    uniform = np.full(len(strategy_red), 1 / len(strategy_red))

    assert studies.nash_gap(payoff_red, payoff_blue, strategy_red, strategy_blue) < 1e-9
    assert studies.nash_gap(payoff_red, payoff_blue, uniform, uniform) > 0.01


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


def test_the_continuous_q_studies_cover_every_encoding():
    assert {k["encoding"] for k, _ in studies.study_continuous_q(range(5))} == {"cartesian", "polar", "square"}
    assert len([k for k, _ in studies.study_continuous_q_tools(range(5))]) == 3
