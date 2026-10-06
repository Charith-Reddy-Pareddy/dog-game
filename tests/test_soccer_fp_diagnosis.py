import pytest

from doggame.soccer import Soccer
from doggame.soccer_pg import train_fictitious_play
from doggame.soccer_fp_diagnosis import exact_fictitious_play

GAME = Soccer(width=4, height=3, goal_size=1)


def test_fictitious_play_with_exact_best_responses_converges():
    _, curve = exact_fictitious_play(GAME, rounds=100, checkpoints=(5, 100))
    early, late = curve[0][1], curve[1][1]
    assert late < 0.3 * early and late < 0.1


def test_the_state_average_option_trains_and_returns_probability_tables():
    average0, average1 = train_fictitious_play(GAME, "reinforce", rounds=2, iterations=2, games=32, mix="state")
    assert average0.shape == (GAME.n_states, 4)
    assert abs(average0.sum(axis=1) - 1).max() < 1e-5


def test_unknown_mix_raises():
    with pytest.raises(ValueError):
        train_fictitious_play(GAME, "reinforce", rounds=1, iterations=1, mix="nonsense")


def test_a_job_reports_how_well_the_best_responses_states_were_trained(monkeypatch):
    import doggame.soccer_fp_diagnosis as diagnosis

    monkeypatch.setitem(diagnosis.VARIANTS, "tiny", (2, 2, "state", True))
    result = diagnosis.run_job((4, 3, 1, "tiny", 0, "reinforce"))

    assert result["adversary_states"] > 0
    assert 0.0 <= result["untrained_share"] <= result["rarely_trained_share"] <= 1.0
    assert len(result["curve"]) >= 1


def test_summary_copes_with_older_results_that_lack_the_coverage_numbers(capsys):
    import doggame.soccer_fp_diagnosis as diagnosis

    old = {"variant": "baseline", "seed": 0, "algorithm": "reinforce", "curve": [[15, 0.9, 0.9]]}
    diagnosis.summarize([old])
    assert "not recorded" in capsys.readouterr().out
