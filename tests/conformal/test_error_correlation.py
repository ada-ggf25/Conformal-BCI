"""Tests for src/conformal/error_correlation.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.error_correlation import error_correlations, weighted_pearson


def test_weighted_pearson_equals_pearson_over_repeated_points():
    """A point of weight w counts exactly like w copies of it."""
    x, y, w = np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.3, 0.8]), np.array([1, 3, 2])
    expected = np.corrcoef(np.repeat(x, w), np.repeat(y, w))[0, 1]
    assert weighted_pearson(x, y, w) == pytest.approx(expected)


def test_groups_are_size_levels_and_equal_count_sigma_bins():
    """Size groups are the distinct |C|/K levels; sigma groups split the windows evenly."""
    rng = np.random.default_rng(0)
    size = rng.integers(1, 5, size=200) / 4
    error = rng.integers(0, 2, size=200).astype(float)
    out = error_correlations(size, error, rng.uniform(0.5, 2.0, 200), n_sigma_bins=5)
    assert out["sigma_points"]["n"].tolist() == [40] * 5
    assert np.all(np.diff(out["sigma_points"]["value"]) > 0)
    assert set(out["size_points"]["value"]) <= {0.25, 0.5, 0.75, 1.0}
    assert out["size_points"]["n"].sum() == 200


def test_error_rising_with_sigma_gives_a_positive_correlation():
    """Wrong windows carry the larger scales, so r(error, sigma) is strongly positive."""
    n = 100
    error = (np.arange(n) >= 50).astype(float)
    out = error_correlations(
        np.full(n, 0.5), error, np.linspace(0.5, 1.5, n), n_sigma_bins=4
    )
    assert out["corr_error_sigma"] > 0.8


@pytest.mark.parametrize(
    ("x", "y", "w"),
    [([1.0], [1.0], [1]), ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1, 1, 1]),
     ([0.0, 1.0], [2.0, 2.0], [3, 4])],
)  # fmt: skip
def test_weighted_pearson_is_undefined_without_two_varying_points(x, y, w):
    """One point, or no spread on either side, gives NaN rather than a fake 0 or 1."""
    assert np.isnan(weighted_pearson(np.array(x), np.array(y), np.array(w)))


def test_weighted_pearson_of_exact_lines_is_plus_or_minus_one():
    """Points on an increasing line correlate +1, on a decreasing line -1, for any weights."""
    x, w = np.array([0.0, 1.0, 3.0]), np.array([5, 1, 2])
    assert weighted_pearson(x, 2 * x + 1, w) == pytest.approx(1.0)
    assert weighted_pearson(x, -x, w) == pytest.approx(-1.0)


def test_weighted_pearson_ignores_a_common_weight_scale():
    """Multiplying every weight by the same factor changes nothing."""
    x, y, w = np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 3.0]), np.array([1, 2, 3])
    assert weighted_pearson(x, y, w) == pytest.approx(weighted_pearson(x, y, 10 * w))


def test_size_points_are_per_level_error_rates():
    """Six windows at two size levels: level 0.25 has errors (0, 0, 1), level 0.5 (1, 1, 0).
    Two points, each of 3 windows, rising error -> r = +1."""
    size = np.array([0.25, 0.5, 0.25, 0.5, 0.25, 0.5])
    error = np.array([0, 1, 0, 1, 1, 0], float)
    out = error_correlations(size, error, np.arange(6.0), n_sigma_bins=2)
    np.testing.assert_allclose(out["size_points"]["value"], [0.25, 0.5])
    np.testing.assert_allclose(out["size_points"]["error_rate"], [1 / 3, 2 / 3])
    assert out["size_points"]["n"].tolist() == [3, 3]
    assert out["corr_error_size"] == pytest.approx(1.0)
    # sigma 0..5 in two bins: (0, 1, 2) errors 0,1,0 and (3, 4, 5) errors 1,1,0
    np.testing.assert_allclose(out["sigma_points"]["value"], [1.0, 4.0])
    np.testing.assert_allclose(out["sigma_points"]["error_rate"], [1 / 3, 2 / 3])


def test_more_bins_than_windows_drops_the_empty_bins():
    """Asking for more sigma bins than windows yields one point per window, no empty points."""
    out = error_correlations(np.ones(3), np.array([0.0, 1.0, 1.0]), np.array([3.0, 1.0, 2.0]),
                             n_sigma_bins=10)  # fmt: skip
    assert out["sigma_points"]["n"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(out["sigma_points"]["value"], [1.0, 2.0, 3.0])
    assert np.isnan(out["corr_error_size"])  # a single size level


def test_group_points_skip_empty_groups():
    """Each non-empty group is one (mean value, error rate, count) point; empty groups vanish."""
    from src.conformal.error_correlation import _group_points

    values, error = np.array([1.0, 3.0, 10.0]), np.array([0.0, 1.0, 1.0])
    pts = _group_points(
        values, error, [np.array([0, 1]), np.array([], int), np.array([2])]
    )
    assert pts["value"].tolist() == [2.0, 10.0]
    assert pts["error_rate"].tolist() == [0.5, 1.0] and pts["n"].tolist() == [2, 1]


def test_group_points_with_no_groups_are_empty():
    """No groups give no points (and so an undefined correlation downstream)."""
    from src.conformal.error_correlation import _group_points

    pts = _group_points(np.zeros(2), np.zeros(2), [])
    assert len(pts["value"]) == len(pts["n"]) == 0
