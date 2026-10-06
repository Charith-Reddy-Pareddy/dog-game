import json

import doggame.soccer_seeds as seeds_module
from doggame.soccer_seeds import load_results, run, run_job, summarize


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


def test_a_rerun_skips_jobs_that_are_already_saved(tmp_path, monkeypatch):
    path = str(tmp_path / "results.jsonl")
    small = dict(seeds=1, self_play_iterations=2, fp_rounds=5, width=4, height=3, goal_size=1, workers=1, results_file=path)

    run(**small)
    assert len(load_results(path)) == 6  # 3 algorithms x 2 methods x 1 seed

    ran = []
    real = seeds_module.run_job
    monkeypatch.setattr(seeds_module, "run_job", lambda job: ran.append(job) or real(job))
    run(**small)

    assert ran == [] and len(load_results(path)) == 6


def test_an_interrupted_run_keeps_what_it_finished(tmp_path):
    path = str(tmp_path / "results.jsonl")
    done = run_job(tiny_job("self-play", 0))
    with open(path, "w") as f:
        f.write(json.dumps(done) + "\n")

    assert load_results(path) == [json.loads(json.dumps(done))]  # JSON turns the curve's tuples into lists
