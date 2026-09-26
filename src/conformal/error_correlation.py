"""Pearson correlation of decoder error with the set size and the scale, over grouped windows.

Per window, decoder error is 0 or 1, so it is averaged within groups of windows first:

- set size: one group per normalized size level ``|C|/K`` (at most ``K`` groups);
- scale: ``n_sigma_bins`` equal-count bins of windows sorted by sigma.

Each group becomes one point (mean value, error rate), and Pearson's r is taken over the points
with each point weighted by its window count. In a session chain every session has its own sigma,
whose overall level carries no meaning, so sigma is passed relative to its session's median.
"""

from __future__ import annotations

import numpy as np

N_SIGMA_BINS = 10
"""Equal-count sigma bins for the scale correlation (deciles)."""


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Pearson's r between ``x`` and ``y`` with each point weighted by ``w``.

    Equal to the ordinary Pearson r over the points repeated ``w`` times. NaN when fewer than
    two points or either side has no spread.
    """
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    if len(x) < 2:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx == 0 or vy == 0:
        return float("nan")
    return float(np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy))


def _group_points(
    values: np.ndarray, error: np.ndarray, groups: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """One (mean value, error rate, count) point per non-empty group of window indices."""
    groups = [g for g in groups if len(g)]
    return {
        "value": np.array([values[g].mean() for g in groups]),
        "error_rate": np.array([error[g].mean() for g in groups]),
        "n": np.array([len(g) for g in groups]),
    }


def error_correlations(
    size_over_k: np.ndarray,
    error: np.ndarray,
    sigma: np.ndarray,
    *,
    n_sigma_bins: int = N_SIGMA_BINS,
) -> dict:
    """Grouped Pearson correlations of decoder error with ``|C|/K`` and with sigma.

    Args:
        size_over_k:  (n,) normalized prediction-set size per test window.
        error:        (n,) decoder error per test window (0 or 1).
        sigma:        (n,) per-window sigma (relative to its session's median in a chain).
        n_sigma_bins: equal-count sigma bins.

    Returns:
        ``{"corr_error_size", "corr_error_sigma", "size_points", "sigma_points"}``. Each
        ``*_points`` is ``{"value", "error_rate", "n"}``, one entry per group.
    """
    size = np.asarray(size_over_k, dtype=float)
    error = np.asarray(error, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    size_points = _group_points(
        size, error, [np.flatnonzero(size == v) for v in np.unique(size)]
    )
    sigma_points = _group_points(
        sigma, error, np.array_split(np.argsort(sigma, kind="stable"), n_sigma_bins)
    )
    return {
        "corr_error_size": weighted_pearson(
            size_points["value"], size_points["error_rate"], size_points["n"]
        ),
        "corr_error_sigma": weighted_pearson(
            sigma_points["value"], sigma_points["error_rate"], sigma_points["n"]
        ),
        "size_points": size_points,
        "sigma_points": sigma_points,
    }
