import json

import pytest


def test_the_trajectory_figure_draws_four_panels_of_the_recorded_paths():
    pytest.importorskip("reportlab.lib.colors")  # skipped where reportlab cannot load its image library
    from doggame.walking_figures import TRAJECTORIES, trajectory_figure

    with open(TRAJECTORIES) as f:
        figure = trajectory_figure(json.load(f))

    # each panel: a frame, a title, two houses, four corner marks and three paths with two end markers each; plus a legend line
    assert figure.width == 440 and len(figure.contents) == 4 * 17 + 1
