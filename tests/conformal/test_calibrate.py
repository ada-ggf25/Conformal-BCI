"""Tests for src/conformal/calibrate.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.calibrate import compute_quantile


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

    @pytest.mark.parametrize(
        ("alpha", "expected"),
        [(0.5, 3.0), (0.2, 5.0), (0.0, np.inf)],
    )
    def test_threshold_is_the_exact_order_statistic(self, alpha, expected):
        """Unsorted scores 1..5 (n = 5): alpha 0.5 -> ceil(3) = 3rd smallest; alpha 0.2 ->
        ceil(4.8) = 5th; alpha 0 -> the 6th, which does not exist."""
        assert compute_quantile(np.array([5.0, 1.0, 4.0, 2.0, 3.0]), alpha) == expected

    def test_marginal_coverage_on_fresh_exchangeable_data(self):
        """For n calibration scores and one exchangeable test score, P(test <= q_hat) is exactly
        ceil((n+1)(1-alpha)) / (n+1): 18/20 = 0.9 for n = 19, alpha = 0.1."""
        rng = np.random.default_rng(0)
        trials = 4000
        hits = 0
        for _ in range(trials):
            s = rng.exponential(size=20)
            hits += s[-1] <= compute_quantile(s[:-1], 0.1)
        assert abs(hits / trials - 0.9) < 0.015  # about 3 binomial standard errors
