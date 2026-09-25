"""Tests for src/conformal/error_correlation.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.error_correlation import error_correlations, weighted_pearson
from src.utils.bundle_readers import error_correlations_for_bundle


def test_weighted_pearson_equals_pearson_over_repeated_points():
    """A point of weight w counts exactly like w copies of it."""
    x, y, w = np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.3, 0.8]), np.array([1, 3, 2])
    expected = np.corrcoef(np.repeat(x, w), np.repeat(y, w))[0, 1]
    assert weighted_pearson(x, y, w) == pytest.approx(expected)


def test_groups_are_size_levels_and_equal_count_sigma_bins():
    """Size groups are the distinct |C|/K levels; sigma groups split the windows evenly."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=200)
    labels = rng.integers(0, 4, size=200)
    scale = rng.uniform(0.5, 2.0, size=200)
    qhats = dict.fromkeys(range(4), 0.8)
    out = error_correlations(probs, scale, labels, qhats, n_sigma_bins=5)
    assert out["sigma_points"]["n"].tolist() == [40] * 5
    assert np.all(np.diff(out["sigma_points"]["value"]) > 0)
    assert set(out["size_points"]["value"]) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert out["size_points"]["n"].sum() == 200


def test_error_rising_with_sigma_gives_a_positive_correlation():
    """Wrong windows carry the larger scales, so r(error, sigma) is strongly positive."""
    n = 100
    labels = np.zeros(n, dtype=int)
    wrong = np.arange(n) >= 50
    probs = np.where(wrong[:, None], [0.2, 0.8], [0.8, 0.2])
    scale = np.linspace(0.5, 1.5, n)
    out = error_correlations(probs, scale, labels, {0: 1.0, 1: 1.0}, n_sigma_bins=4)
    assert out["corr_error_sigma"] > 0.8


def test_only_test_windows_are_read(tmp_path):
    """A non-test window with an extreme error must not leak into the correlations."""
    arr_path = tmp_path / "arrays.npz"
    np.savez(
        arr_path,
        win_split=np.array([0, 2, 2, 2, 2, 2, 2, 2, 2]),  # index 0 is train
        win_probs=np.vstack([[0.0, 1.0]] + [[0.9, 0.1]] * 8),
        win_sigma=np.ones(9),
        win_region=np.zeros(9, dtype=int),
        win_session_idx=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]),
    )
    bundle = {
        "n_classes": 2,
        "_arrays_path": arr_path,
        "conformal": {
            "arms": {
                "learned-scale class-conditional": {
                    "q_hat_by_region": {"0": 1.0, "1": 1.0},
                }
            }
        },
    }
    got = error_correlations_for_bundle(bundle)
    assert got["size_points"]["n"].sum() == 8
    assert got["size_points"]["error_rate"].max() == 0.0  # every test window is correct
