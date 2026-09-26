"""Tests for src/conformal/calibrate.py."""

from __future__ import annotations

import numpy as np

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
