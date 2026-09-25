"""Conformal evaluation for the per-maze pipeline.

This module owns the evaluation math the notebook uses to: pool split
indices across sessions, compute the sigma-scaled per-region quantile, and
retrain the scale across a coverage sweep to trace the method's achievable
frontier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .calibrate import class_conditional_quantiles_with_fallback
from .learned import scale_of, train_scale
from .predict import class_conditional_prediction_sets, empirical_coverage

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_offsets(session_data: dict, valid_sessions: list[str]) -> dict[str, int]:
    """Row offset of each session in the pooled per-window arrays."""
    off, pos = {}, 0
    for s in valid_sessions:
        off[s] = pos
        pos += len(session_data[s]["regions"])
    return off


def pool_indices(
    session_data: dict,
    valid_sessions: list[str],
    key: str,
    offsets: dict[str, int],
) -> np.ndarray:
    """Pool one per-session split block into global row indices of the pooled arrays.

    Sorts each session's block and shifts it by the session's offset.
    """
    return np.concatenate(
        [np.sort(session_data[s][key]) + offsets[s] for s in valid_sessions]
    )


def block_layout(
    session_data: dict,
    valid_sessions: list[str],
    index_key: str,
) -> np.ndarray:
    """Per-window session index for one split block."""
    sess = []
    for si, s in enumerate(valid_sessions):
        n = len(session_data[s][index_key])
        if n:
            sess.append(np.full(n, si))
    return np.concatenate(sess)


def class_conditional_q_scaled(
    probs: np.ndarray,
    labels: np.ndarray,
    sigma: np.ndarray,
    val_i: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict[int, float]:
    """Per-region q_hat from the sigma-scaled true-class scores on the val block.

    ``q_hat_c`` is the ``(1-alpha)`` quantile of ``(1 - p_true)/sigma`` within
    region ``c``. A region with too few val windows falls back to the global q_hat.
    """
    scaled_true_val = (1.0 - probs[val_i, labels[val_i]]) / sigma[val_i]
    return class_conditional_quantiles_with_fallback(
        scaled_true_val, labels[val_i], n_classes, alpha
    )


def retrain_coverage_sweep(
    signals_z: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    fit_i: np.ndarray,
    val_i: np.ndarray,
    test_i: np.ndarray,
    n_classes: int,
    targets: Sequence[float],
    *,
    hidden: int,
    depth: int,
    epochs: int,
    tau: float,
    kappa: float,
    early_stop_patience: int | None,
    seed: int,
    device: str = "cpu",
    size_weight: float = 1.0,
) -> list[dict]:
    """Coverage-sharpness curve where the scale is retrained at each target.

    Args:
        signals_z: (n, n_signals) standardized per-window signals, pooled.
        probs:     (n, n_classes) softmax probabilities, pooled.
        labels:    (n,) true region per pooled window.
        fit_i:     indices of the block the scale is trained on.
        val_i:     indices of the block used for early stopping and q_hat.
        test_i:    indices of the block the curve is measured on.
        n_classes: total number of regions.
        targets:   coverage targets, e.g. ``[0.80, 0.90]``. Not alphas. each is
                   converted internally as ``alpha = 1 - target``.

    Returns:
        One row per target, in ``targets`` order, ``{target, coverage, size}``.
        The target asked for, the coverage achieved on the test block, and the mean
        prediction-set size there.
    """
    y_te = labels[test_i]
    rows = []
    for tgt in targets:
        a = 1.0 - float(tgt)
        net, _ = train_scale(
            signals_z[fit_i],
            probs[fit_i],
            labels[fit_i],
            alpha=a,
            hidden=hidden,
            depth=depth,
            epochs=epochs,
            tau=tau,
            kappa=kappa,
            size_weight=size_weight,
            val_signals_z=signals_z[val_i],
            val_probs=probs[val_i],
            val_labels=labels[val_i],
            early_stop_patience=early_stop_patience,
            n_classes=n_classes,
            seed=seed,
            device=device,
        )
        sig = scale_of(net, signals_z, device=device)
        scaled_val = (1.0 - probs[val_i, labels[val_i]]) / sig[val_i]
        q = class_conditional_quantiles_with_fallback(
            scaled_val, labels[val_i], n_classes, a
        )
        sets = class_conditional_prediction_sets(probs[test_i], sig[test_i], q)
        rows.append(
            {
                "target": float(tgt),
                "coverage": float(empirical_coverage(sets, y_te)),
                "size": float(np.mean([len(s) for s in sets])),
            }
        )
    return rows
