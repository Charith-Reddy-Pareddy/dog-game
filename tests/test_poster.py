import numpy as np
import pytest

from doggame.experiments import DEFAULT_CONFIGS
from doggame.poster import rho, slack


def test_the_slack_needed_rises_with_inertia_and_falls_with_patience():
    w0 = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(rho(w0, 0.9)) > 0) and np.all(rho(w0, 0.99) < rho(w0, 0.5))
    assert rho(0.0, 0.9) == 0 and np.isclose(rho(0.6, 0.0), 0.6)


def test_each_setup_has_the_slack_of_its_tightest_player():
    assert [round(slack(c), 3) for c in DEFAULT_CONFIGS] == [0.6, 0.0, 0.0, 0.9, 0.2]


def test_the_poster_builds(tmp_path):
    pytest.importorskip("reportlab")
    from doggame.poster import build

    data = open(build(str(tmp_path / "poster.pdf")), "rb").read()
    assert data.startswith(b"%PDF") and len(data) > 100_000
