"""Decoder evaluation for pooled per-maze runs."""

from __future__ import annotations

import numpy as np


def decoder_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> dict:
    """Read of a decoder: the reported accuracy plus the baselines that interpret it.

    Args:
        y_true:    (n,) true region labels for the block being read (e.g. pooled test).
        probs:     (n, n_classes) softmax probabilities; the prediction is ``argmax``.
        n_classes: number of regions K.

    Returns:
        dict with ``raw_accuracy``, ``chance`` (1/K) and ``majority_baseline`` (the most
        frequent region's share of ``y_true``).
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(probs).argmax(1)
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "raw_accuracy": float((pred == y_true).mean()),
        "chance": float(1.0 / n_classes),
        "majority_baseline": float(counts.max() / counts.sum()),
    }
