"""Tests for src/conformal/trajectory.py."""

from __future__ import annotations

import numpy as np

from src.utils.bundle_readers import trajectory_summary_for_bundle


def test_only_test_windows_are_read(tmp_path):
    """A non-test window with an extreme error must not leak into the trajectory."""
    arr_path = tmp_path / "arrays.npz"
    np.savez(
        arr_path,
        win_split=np.array([0, 2, 2, 2, 2, 2, 2, 2, 2]),  # index 0 is train
        win_probs=np.vstack([[0.0, 1.0]] + [[0.9, 0.1]] * 8),
        win_sigma=np.ones(9),
        win_region=np.zeros(9, dtype=int),
        win_session_idx=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]),
    )
    bundle = {
        "n_classes": 2,
        "_arrays_path": arr_path,
        "conformal": {
            "arms": {
                "learned-scale class-conditional": {
                    "q_hat_by_region": {"0": float("inf"), "1": float("inf")},
                }
            }
        },
    }
    got = trajectory_summary_for_bundle(bundle, bin_width_sec=2.0, min_sessions=1)
    assert got["error_max"] == 0.0  # every test window is correct
