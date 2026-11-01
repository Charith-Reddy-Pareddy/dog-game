import numpy as np
import pytest

from doggame.continuous_br import bisect_best_response, gradient_best_response, quadratic_best_response

METHODS = [bisect_best_response, gradient_best_response, quadratic_best_response]


@pytest.mark.parametrize("method", METHODS)
def test_finds_the_peak_of_a_parabola(method):
    q = lambda a: -(a - 1.3) ** 2
    assert method(q, 0.0, 2 * np.pi) == pytest.approx(1.3, abs=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_stops_at_the_edge_when_the_peak_is_outside(method):
    q = lambda a: a  # always better to go further
    assert method(q, 0.0, 2.0) == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("method", [bisect_best_response, gradient_best_response])
def test_slope_methods_find_a_non_parabolic_peak(method):
    q = lambda a: -(a - 2.0) ** 4
    assert method(q, 0.0, 4.0) == pytest.approx(2.0, abs=0.05)


def test_quadratic_fit_is_well_conditioned_when_the_best_sample_is_at_an_edge():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # numpy warns if the fit is underdetermined
        quadratic_best_response(lambda a: a, 0.0, 2.0)
