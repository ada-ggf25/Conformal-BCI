"""Tests for src/conformal/error_correlation.py."""

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
