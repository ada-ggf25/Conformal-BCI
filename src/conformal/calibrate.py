"""Calibration functions: compute the score threshold q_hat from a held-out set.

Original to this project.
"""

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


def class_conditional_quantiles_with_fallback(
    scores: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict[int, float]:
    """Class-conditional q_hat with a global-q_hat fallback for data-starved regions.

    A region needs at least ``ceil(1/alpha) - 1`` calibration scores to support its own finite
    (1-alpha) quantile. Below that, a plain per-region quantile is +inf, admitting the whole
    label space for that region. Here a starved region instead falls back to the global-q_hat.
    Still a valid threshold, and the production fix for low-data mazes.

    Args:
        scores:    (n,) non-conformity scores on the calibration block. For a learned scale
                   these are already sigma-scaled: (1 - p_hat_y) / sigma(x).
        labels:    (n,) true region index per calibration window.
        n_classes: total number of regions.
        alpha:     miscoverage level.

    Returns:
        dict {region: q_hat} with every region in range(n_classes) present. A region that
        took the fallback keeps marginal validity but no longer carries its own
        class-conditional coverage guarantee.
    """
    floor = int(np.ceil(1.0 / alpha)) - 1
    q_global = compute_quantile(scores, alpha)
    out: dict[int, float] = {}
    for c in range(n_classes):
        sc = scores[labels == c]
        out[c] = compute_quantile(sc, alpha) if len(sc) > floor else q_global
    return out
