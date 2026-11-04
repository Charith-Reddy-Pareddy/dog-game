import numpy as np

from doggame import dog_game_summary as S
from doggame import summary_figures as F


def test_figure_5_gives_the_skewed_target_numbers_quoted_in_the_summary(tmp_path):
    diff = F.fig5(tmp_path / "fig5.png", S.load(S.STUDIES))
    assert len(diff) == 60 and (diff < 0).sum() == 46 and round(-diff.mean(), 3) == 0.008
    assert round(diff.std(ddof=1) / np.sqrt(len(diff)), 4) == 0.0025


def test_figure_8_times_each_receiver_and_the_paper_figures_are_vector_pdfs(tmp_path):
    assert F.fig8(tmp_path / "fig8.png") == [27, 11, 10, 7]  # averaging, median, compensated averaging, planner
    out = F.paper(tmp_path / "paper")
    assert sorted(p.name for p in out.glob("*.pdf")) == ["compensated_paths.pdf", "design.pdf", "reversal.pdf", "weight.pdf"]


def test_the_dynamic_game_figures_show_exact_arrival_only_where_the_receiver_can_reach_its_end(tmp_path):
    # averaging: x has no pivot and only approaches 1/3, y has one and lands in round 5; the median lands in both
    assert F.fig11(tmp_path / "fig11.png") == [(None, 5), (6, 3)]
    assert F.fig12(tmp_path / "fig12.png") == [2, 3, 3, 15, 20, 10]  # averaging reports stay near the edges, median ones move
