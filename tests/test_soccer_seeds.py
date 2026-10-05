from doggame.soccer_seeds import run_job, summarize


def tiny_job(method, seed):
    return (4, 3, 1, "reinforce", method, seed, 4, 2)  # a 4x3 board, 4 iterations or rounds, a checkpoint every 2


def test_a_job_scores_the_policy_and_records_progress():
    for method in ("self-play", "fictitious play"):
        result = run_job(tiny_job(method, seed=0))
        assert set(result["nash"]) == {"wins", "losses", "ties"}
        assert sum(result["nash"].values()) == 1000
        assert [step for step, _ in result["curve"]] == [2, 4]


def test_the_same_seed_gives_the_same_result():
    assert run_job(tiny_job("self-play", 1)) == run_job(tiny_job("self-play", 1))


def test_summary_prints_a_row_per_method(capsys):
    summarize([run_job(tiny_job("self-play", s)) for s in range(2)])
    out = capsys.readouterr().out
    assert "reinforce self-play" in out and "2" in out
