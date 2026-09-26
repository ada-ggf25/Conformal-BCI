"""Calibration functions: compute the score threshold q_hat from a held-out set."""

from __future__ import annotations

import numpy as np


def compute_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with finite-sample coverage guarantee.

    q̂ = the ⌈(n+1)(1-α)⌉-th order statistic of scores.

    Returns +inf when the calibration set is empty or too small to achieve
    (1-α) coverage (⌈(n+1)(1-α)⌉ > n).

    Args:
        scores: (n,) non-conformity scores from the calibration set. Need not be sorted.
        alpha:  miscoverage level, e.g. 0.1 for 90 % target coverage.

    Returns:
        q_hat: float threshold on the same scale as ``scores``, or +inf if the set is
        too small. A score at or below q_hat admits its class into the prediction set.
    """
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])
