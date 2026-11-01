import numpy as np

from doggame import dog_game_summary as S
from doggame import summary_figures as F


def test_figure_2_gives_the_round_counts_quoted_in_its_caption(tmp_path):
    assert F.fig2(tmp_path / "fig2.png") == [27, 11]  # averaging, median
    assert (tmp_path / "fig2.png").stat().st_size > 10_000


def test_figure_5_gives_the_skewed_target_numbers_quoted_in_the_summary(tmp_path):
    diff = F.fig5(tmp_path / "fig5.png", S.load(S.STUDIES))
    assert len(diff) == 60 and (diff < 0).sum() == 46 and round(-diff.mean(), 3) == 0.008
    assert round(diff.std(ddof=1) / np.sqrt(len(diff)), 4) == 0.0025
