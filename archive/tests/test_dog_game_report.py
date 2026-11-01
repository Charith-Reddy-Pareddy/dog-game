import json

from doggame.dog_game_report import describe, load
from doggame.dog_game_study import ARCHITECTURES, run_convergence, run_walking
from doggame.experiments import DEFAULT_CONFIGS


def finished_runs(path, seed):
    """A results file where every run for this seed is already done."""
    rows = [dict(kind="convergence", architecture=a, seed=seed, config=i, exploitability=0.001)
            for a in ARCHITECTURES for i in range(len(DEFAULT_CONFIGS))]
    rows += [dict(kind="walking", architecture=m, seed=seed, config=0, before=1.0, after=0.1) for m in ("reinforce", "nash-q")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_finished_runs_are_not_run_again(tmp_path):
    path = tmp_path / "runs.jsonl"
    finished_runs(path, seed=0)
    before = path.read_text()

    run_convergence([0], str(path))
    run_walking([0], str(path))

    assert path.read_text() == before


def test_load_splits_convergence_from_walking_runs(tmp_path):
    path = tmp_path / "runs.jsonl"
    finished_runs(path, seed=3)

    convergence, walking = load(str(path))

    assert len(convergence) == len(ARCHITECTURES) * len(DEFAULT_CONFIGS) and len(walking) == 2
    assert {r["seed"] for r in convergence + walking} == {3}


def test_describe_names_both_houses_and_the_weight():
    text = describe(dict(house_red=(0.9, 0.2), house_blue=(0.1, 0.8), w=0.5))
    assert "red (0.9, 0.2)" in text and "blue (0.1, 0.8)" in text and "w = 0.5" in text
