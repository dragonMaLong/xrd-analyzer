import numpy as np
import pytest

from xrd_analyzer.core.analysis import calculate_rfit_percent


def test_rfit_percent_matches_relative_l2_residual():
    observed = np.array([0.0, 1.0, 2.0, 1.0])
    calculated = np.array([0.0, 0.9, 1.8, 1.1])

    expected = 100.0 * np.linalg.norm(observed - calculated) / np.linalg.norm(observed)

    assert calculate_rfit_percent(observed, calculated) == pytest.approx(expected)
    assert calculate_rfit_percent(observed * 25.0, calculated * 25.0) == pytest.approx(expected)


def test_rfit_percent_handles_invalid_input_without_false_perfect_fit():
    assert np.isnan(calculate_rfit_percent([0.0, 0.0], [0.0, 0.0]))
    assert calculate_rfit_percent([1.0, np.nan], [0.9, 4.0]) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        calculate_rfit_percent([1.0], [1.0, 2.0])
