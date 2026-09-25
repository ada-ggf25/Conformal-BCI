"""Decoder error, scale and set size vs. time since calibration."""

from __future__ import annotations

import numpy as np

from .predict import class_conditional_prediction_sets


def binned_group_weighted_mean(
    values: np.ndarray, bin_idx: np.ndarray, group_idx: np.ndarray, n_bins: int
) -> np.ndarray:
    """Group-weighted mean per position bin: average the per-group means.

    Every group counts once, so a long chunk does not outweigh a short one. The group here is
    the session, which is the statistical unit everywhere in this project: windows within a
    session are autocorrelated, so averaging raw windows would let the longest recordings
    dominate a bin. ``group_idx`` is an explicit argument rather than a closed-over global so
    that choice stays visible at the call site.

    Returns a length-``n_bins`` array (NaN for empty bins).
    """
    values = np.asarray(values, float)
    bin_idx = np.asarray(bin_idx)
    group_idx = np.asarray(group_idx)
    groups = np.unique(group_idx)
    out = []
    for b in range(n_bins):
        in_bin = bin_idx == b
        per = [
            values[m].mean() for i in groups if (m := in_bin & (group_idx == i)).any()
        ]
        out.append(np.mean(per) if per else np.nan)
    return np.array(out)


def calibration_distance_trajectory(
    probs: np.ndarray,
    scale: np.ndarray,
    labels: np.ndarray,
    session_idx: np.ndarray,
    qhats: dict[int, float],
    *,
    bin_width_sec: float = 20.0,
    seconds_per_window: float = 1.0,
    min_sessions: int = 3,
) -> dict[str, np.ndarray]:
    """Binned trajectory of error, sigma, and set size vs. time since calibration.

    Pure inference over an already-fitted sigma/qhats (no retraining). Each window's x-value
    is its distance in seconds from the end of its session's calibration (val) block. Session
    lengths are not normalized to a shared 0->1 axis, so a bin far from calibration is only
    reachable by sessions that ran that long.

    Args:
        probs:               (n, n_classes) softmax probabilities, test windows only.
        scale:               (n,) strictly positive per-window sigma, test windows only.
        labels:              (n,) true class index, test windows only.
        session_idx:         (n,) session id per window, a session's rows must already be
                             contiguous and chronological.
        qhats:               {class_index: q_hat} class-conditional thresholds.
        bin_width_sec:       width of each time-since-calibration bin, in seconds.
        seconds_per_window:  real seconds spanned by one window.
        min_sessions:        minimum contributing sessions for a bin to be kept.

    Returns:
        ``{"bin_center", "n_sessions", "error", "sigma", "size_over_k"}``: all the same
        length, the surviving prefix of bins (possibly empty if even bin 0 fails the
        ``min_sessions`` gate).
    """
    n_classes = probs.shape[1]
    sets = class_conditional_prediction_sets(probs, scale, qhats)
    error = (probs.argmax(1) != np.asarray(labels)).astype(float)
    size = np.array([len(s) for s in sets], dtype=float)

    session_idx = np.asarray(session_idx)
    pos_sec = np.zeros(len(session_idx), dtype=float)
    for s in np.unique(session_idx):
        m = session_idx == s
        n = int(m.sum())
        pos_sec[m] = np.arange(n) * seconds_per_window

    n_bins_full = int(np.floor(pos_sec.max() / bin_width_sec)) + 1
    bin_idx = np.clip((pos_sec / bin_width_sec).astype(int), 0, n_bins_full - 1)
    bin_center = (np.arange(n_bins_full) + 0.5) * bin_width_sec

    n_sessions = np.array(
        [len(np.unique(session_idx[bin_idx == b])) for b in range(n_bins_full)]
    )
    keep = n_bins_full
    for b in range(n_bins_full):
        if n_sessions[b] < min_sessions:
            keep = b
            break

    err_bin = binned_group_weighted_mean(error, bin_idx, session_idx, n_bins_full)
    sig_bin = binned_group_weighted_mean(
        np.asarray(scale, dtype=float), bin_idx, session_idx, n_bins_full
    )
    size_bin = binned_group_weighted_mean(
        size / n_classes, bin_idx, session_idx, n_bins_full
    )

    sl = slice(0, keep)
    return {
        "bin_center": bin_center[sl],
        "n_sessions": n_sessions[sl],
        "error": err_bin[sl],
        "sigma": sig_bin[sl],
        "size_over_k": size_bin[sl],
    }


def trajectory_summary(traj: dict[str, np.ndarray]) -> dict[str, float]:
    """Collapse a ``calibration_distance_trajectory`` into one summary row.

    Args:
        traj: the dict ``calibration_distance_trajectory`` returns.

    Returns:
        ``{"error_min", "error_max", "size_over_k_min", "size_over_k_max", "corr_error_size",
        "corr_error_sigma"}``.
    """
    error = np.asarray(traj["error"], dtype=float)
    size = np.asarray(traj["size_over_k"], dtype=float)
    sigma = np.asarray(traj["sigma"], dtype=float)
    nan = float("nan")
    if len(error) == 0:
        return {
            "error_min": nan,
            "error_max": nan,
            "size_over_k_min": nan,
            "size_over_k_max": nan,
            "corr_error_size": nan,
            "corr_error_sigma": nan,
        }

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return nan
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "error_min": float(error.min()),
        "error_max": float(error.max()),
        "size_over_k_min": float(size.min()),
        "size_over_k_max": float(size.max()),
        "corr_error_size": _corr(error, size),
        "corr_error_sigma": _corr(error, sigma),
    }
