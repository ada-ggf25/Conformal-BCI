"""Prediction-set construction plus the coverage and sharpness metrics that read it."""

from __future__ import annotations

import numpy as np


def class_conditional_prediction_sets(
    probs: np.ndarray, scale: np.ndarray, qhats: dict[int, float]
) -> list[list[int]]:
    """Class-conditional prediction sets on a sigma-scaled score.

    C(x): each class is admitted using its own calibrated threshold applied to the scaled
    non-conformity score. Combines ``scale_of`` (the trained shared σ) with per-region
    ``qhats`` from ``class_conditional_quantiles_with_fallback`` computed on the
    scaled calibration scores.

    Args:
        probs: (n, n_classes) softmax probabilities.
        scale: (n,) strictly positive per-window scale σ(x).
        qhats: dict {class_index: q_hat}, every class in ``range(n_classes)`` required.

    Returns:
        n lists of class indices, one per window. A list may be empty when no class
        clears its threshold.
    """
    p = np.asarray(probs, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    if s.shape[0] != p.shape[0]:
        raise ValueError(
            f"scale (n={s.shape[0]}) and probs (n={p.shape[0]}) must align"
        )
    n_classes = p.shape[1]
    scaled = (1.0 - p) / s[:, None]  # (n, n_classes)
    q = np.array([qhats[c] for c in range(n_classes)], dtype=np.float64)
    return [
        [int(c) for c in range(n_classes) if scaled[i, c] <= q[c]]
        for i in range(len(p))
    ]


def empirical_coverage(sets: list[list[int]], labels: np.ndarray) -> float:
    """Fraction of samples whose true label falls inside the prediction set."""
    hits = sum(int(lbl) in s for s, lbl in zip(sets, labels, strict=False))
    return hits / len(labels) if len(labels) > 0 else 0.0


def _mean_by_class(
    values: np.ndarray,
    sets: list[list[int]],
    labels: np.ndarray,
    n_classes: int,
    min_count: int,
) -> dict[int, float]:
    """Mean of a per-window quantity, grouped by true label, NaN below ``min_count``.

    The shared body of ``coverage_by_class`` and ``size_by_class``, so the two read
    their per-region statistics off the same windows under the same gate.
    """
    labels = np.asarray(labels)
    if len(sets) != labels.shape[0]:
        raise ValueError(
            f"sets (n={len(sets)}) and labels (n={labels.shape[0]}) must align"
        )
    out: dict[int, float] = {}
    for c in range(n_classes):
        mask = labels == c
        count = int(mask.sum())
        out[c] = (
            float(values[mask].mean()) if count >= max(min_count, 1) else float("nan")
        )
    return out


def coverage_by_class(
    sets: list[list[int]],
    labels: np.ndarray,
    n_classes: int,
    min_count: int = 1,
) -> dict[int, float]:
    """Class-conditional empirical coverage, split by true label.

    ``coverage_c`` = fraction of samples whose true label is ``c`` that have
    ``c`` inside their prediction set ``C(x)``.

    Returns:
        dict {class_index: coverage in [0, 1]}: classes with < ``min_count`` true
        samples map to NaN. Every class in ``range(n_classes)`` is present.
    """
    hits = np.array(
        [int(lbl) in s for s, lbl in zip(sets, labels, strict=False)], dtype=bool
    )
    return _mean_by_class(hits, sets, labels, n_classes, min_count)


def size_by_class(
    sets: list[list[int]],
    labels: np.ndarray,
    n_classes: int,
    min_count: int = 1,
) -> dict[int, float]:
    """Mean prediction-set size split by true label.

    Reading this beside ``coverage_by_class`` on the same
    ``min_count`` gate gives the per-region coverage/sharpness pair.

    Returns:
        dict {class_index: mean |C(x)| over samples of that true class}: classes with
        < ``min_count`` true samples map to NaN. Every class in ``range(n_classes)``
        is present.
    """
    sizes = np.array([len(s) for s in sets], dtype=float)
    return _mean_by_class(sizes, sets, labels, n_classes, min_count)
