import numpy as np

from doggame import dog_game_summary as S
from doggame import mechanisms as M
from doggame import summary_figures as F


def test_figure_5_gives_the_skewed_target_numbers_quoted_in_the_summary(tmp_path):
    diff = F.fig5(tmp_path / "fig5.png", S.load(S.STUDIES))
    assert len(diff) == 60 and (diff < 0).sum() == 46 and round(-diff.mean(), 3) == 0.008
    assert round(diff.std(ddof=1) / np.sqrt(len(diff)), 4) == 0.0025


def test_figure_8_times_each_receiver_and_the_paper_figures_are_vector_pdfs(tmp_path):
    assert F.fig8(tmp_path / "fig8.png") == [27, 11, 10, 7]  # averaging, median, compensated averaging, planner
    out = F.paper(tmp_path / "paper")
    assert sorted(p.name for p in out.glob("*.pdf")) == ["compensated_paths.pdf", "design.pdf", "reversal.pdf", "weight.pdf"]


def test_the_dynamic_game_figures_show_exact_arrival_only_where_a_target_decides(tmp_path):
    # x ends on the phantom 1/2 and never arrives, y lands on influencer 1's target in round 4 (Proposition 18)
    assert F.fig11(tmp_path / "fig11.png") == (None, 4)
    assert F.fig12(tmp_path / "fig12.png") == [6, 2]
    land, left = F.fig13(tmp_path / "fig13.png")
    assert land == 4 and round(left, 4) == 0.0014


def test_the_game_of_the_dynamic_figures_is_an_equilibrium_in_both_coordinates():
    aims, _ = F.two_influencers(0.8)
    for c in range(2):
        policy = lambda x, c=c: M.reports("mean", x, aims[c], F.TWO_WEIGHTS, 0.8)
        assert M.deviation_gain("mean", policy, F.TWO_TARGETS[:, c], F.TWO_WEIGHTS, 0.8) < 1e-6
