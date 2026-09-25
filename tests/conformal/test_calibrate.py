"""Tests for src/conformal/calibrate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.calibrate import (
    class_conditional_quantiles_with_fallback,
    compute_quantile,
)

N_CLASSES = 6


class TestComputeQuantile:
    """The `+inf` returns are not defensive coding. Admitting the whole label space
    is the only answer that preserves the coverage guarantee when the calibration set
    cannot support it."""

    def test_empty_scores_returns_inf(self):
        """An empty calibration set supports no finite threshold, so q_hat is +inf."""
        assert compute_quantile(np.array([]), 0.1) == np.inf

    def test_too_small_calibration_set_returns_inf(self):
        """Below the finite-sample floor no order statistic reaches the target, so q_hat is +inf."""
        # n=5, alpha=0.1: ceil(6*0.9)=ceil(5.4)=6 > 5 -> +inf
        assert compute_quantile(np.linspace(0, 1, 5), 0.1) == np.inf

    def test_coverage_guarantee_on_calibration_set(self):
        """Calibrating and evaluating on the same set must give coverage >= 1-alpha."""
        scores = np.random.RandomState(42).uniform(0, 1, 500)
        for alpha in [0.05, 0.1, 0.2]:
            q = compute_quantile(scores, alpha)
            empirical = (scores <= q).mean()
            assert empirical >= 1 - alpha, (
                f"alpha={alpha}: coverage {empirical:.4f} < {1 - alpha:.4f}"
            )


class TestClassConditionalQuantilesWithFallback:
    def test_starved_region_falls_back_to_global(self):
        """A region with too few cal scores gets the marginal q_hat, not +inf. This is
        the production fix for high-K/low-data mazes."""
        alpha = 0.1
        # region 0 well populated, region 1 has one score (below the finite-sample floor)
        scores = np.concatenate([np.linspace(0.0, 1.0, 100), [0.5]])
        labels = np.array([0] * 100 + [1])
        q = class_conditional_quantiles_with_fallback(scores, labels, 2, alpha)
        assert np.isfinite(q[1])
        assert q[1] == pytest.approx(compute_quantile(scores, alpha))

    def test_well_populated_matches_the_plain_per_region_quantile(self):
        """Above the floor the fallback never fires, so the result must equal the plain
        per-region quantile."""
        rng = np.random.RandomState(3)
        scores, labels = rng.rand(600), rng.randint(0, N_CLASSES, size=600)
        q_fb = class_conditional_quantiles_with_fallback(scores, labels, N_CLASSES, 0.1)
        for c in range(N_CLASSES):
            assert q_fb[c] == pytest.approx(compute_quantile(scores[labels == c], 0.1))
