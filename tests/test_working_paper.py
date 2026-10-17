from doggame import dog_game_summary as S
from doggame.working_paper import build, counts


def test_the_paper_quotes_the_random_game_counts_from_the_results(tmp_path):
    rows = [dict(study="inertia_random", key=dict(game=0), kind="straddle", gain=0.0, holds=True),
            dict(study="inertia_random", key=dict(game=1), kind="interior", gain=0.01, holds=False)]
    assert counts(rows) == dict(straddle="1", interior="1", interior_ok="0", holds="1")
    text = open(build(str(tmp_path / "paper.md"))).read()
    assert "Proposition 2 holds in all" in text and "## References" in text
    assert counts(S.load(S.STUDIES))["straddle"] in text
