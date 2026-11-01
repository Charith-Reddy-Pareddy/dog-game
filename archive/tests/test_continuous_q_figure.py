from doggame import dog_game_summary as S
from doggame.continuous_q_figure import build


def test_the_trajectory_figure_is_drawn_from_the_saved_runs(tmp_path):
    assert {r["key"]["encoding"] for r in S.study(S.load(S.STUDIES), "continuous_q_tools")} >= {"cartesian", "polar", "square"}
    out = build(str(tmp_path / "paths.svg"))
    assert "<svg" in open(out).read(2000)
