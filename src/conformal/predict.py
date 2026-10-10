"""Prediction-set construction plus the coverage and sharpness metrics that read it."""

from __future__ import annotations

import numpy as np


def class_conditional_prediction_sets(
    probs: np.ndarray, scale: np.ndarray, qhats: dict[int, float]
) -> list[list[int]]:
    """Class-conditional prediction sets on a sigma-scaled score.

    C(x): each class is admitted using its own calibrated threshold applied to the scaled
    non-conformity score. Combines ``scale_of`` (the trained shared σ) with per-region
    ``qhats`` from ``src.chain.pool.pooled_qhats`` computed on the
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
