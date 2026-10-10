"""Evaluating one session's test block under a set of per-region thresholds."""

from __future__ import annotations

import numpy as np

from ..conformal.predict import class_conditional_prediction_sets


def evaluate_block(
    test: dict[str, np.ndarray],
    qhats: dict[int, float],
    levels: dict[int, str],
    n_classes: int,
) -> dict[str, np.ndarray]:
    """Per-window results on a scored test block.

    Args:
        test:      ``{"probs", "sigma", "labels"}`` for the test windows.
        qhats:     ``{region: q_hat}``.
        levels:    ``{region: level}``, recorded per window by its true region.
        n_classes: number of regions K.

    Returns:
        ``{"covered", "size_over_k", "error", "level"}``, one entry per test window.
    """
    sets = class_conditional_prediction_sets(test["probs"], test["sigma"], qhats)
    y = test["labels"]
    return {
        "covered": np.array([int(y[i]) in s for i, s in enumerate(sets)], dtype=float),
        "size_over_k": np.array([len(s) for s in sets], dtype=float) / n_classes,
        "error": (test["probs"].argmax(1) != y).astype(float),
        "level": np.array([levels[int(c)] for c in y]),
    }
