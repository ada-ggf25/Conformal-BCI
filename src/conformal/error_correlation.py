"""Pearson correlation of decoder error with the set size and the scale, over grouped windows.

Per window, decoder error is 0 or 1, so it is averaged within groups of windows first:

- set size: one group per normalized size level ``|C|/K`` (at most ``K`` groups);
- scale: ``n_sigma_bins`` equal-count bins of windows sorted by sigma.

Each group becomes one point (mean value, error rate), and Pearson's r is taken over the points
with each point weighted by its window count.
"""

from __future__ import annotations

import numpy as np

from .predict import class_conditional_prediction_sets

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
    probs: np.ndarray,
    scale: np.ndarray,
    labels: np.ndarray,
    qhats: dict[int, float],
    *,
    n_sigma_bins: int = N_SIGMA_BINS,
) -> dict:
    """Grouped Pearson correlations of decoder error with ``|C|/K`` and with sigma.

    Args:
        probs:        (n, n_classes) softmax probabilities, test windows only.
        scale:        (n,) strictly positive per-window sigma, test windows only.
        labels:       (n,) true class index, test windows only.
        qhats:        {class_index: q_hat} class-conditional thresholds.
        n_sigma_bins: equal-count sigma bins.

    Returns:
        ``{"corr_error_size", "corr_error_sigma", "size_points", "sigma_points"}``. Each
        ``*_points`` is ``{"value", "error_rate", "n"}``, one entry per group.
    """
    probs = np.asarray(probs)
    scale = np.asarray(scale, dtype=float)
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    sets = class_conditional_prediction_sets(probs, scale, qhats)
    size = np.array([len(s) for s in sets], dtype=float) / n_classes
    error = (probs.argmax(1) != labels).astype(float)

    size_points = _group_points(
        size, error, [np.flatnonzero(size == v) for v in np.unique(size)]
    )
    sigma_points = _group_points(
        scale, error, np.array_split(np.argsort(scale, kind="stable"), n_sigma_bins)
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
