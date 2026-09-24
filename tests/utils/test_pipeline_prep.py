"""Tests for src/utils/pipeline_prep.py."""

from __future__ import annotations

import numpy as np

from src.utils.pipeline_prep import force_session_specific, normalize_session_windows


def _session(n=10, t=2, n_neurons=3, n_train=6):
    """Return a session whose held-out rows are offset from the train block."""
    rng = np.random.default_rng(0)
    # train block deliberately offset from the rest so wrong rows show up
    w = rng.normal(loc=5.0, scale=1.0, size=(n, t, n_neurons))
    w[n_train:] += 50.0
    return {
        "windows": w.astype("float32"),
        "train_idx": np.arange(n_train),
        "test_idx": np.arange(n_train, n),
    }


class TestNormalizeSessionWindows:
    def test_statistics_come_from_train_rows_only(self):
        """The held-out block keeps its offset, that is the point of train-only
        stats. Fitting on all rows would leak the test distribution into training and
        inflate every downstream number."""
        d = {"s": _session()}
        normalize_session_windows(d, ["s"])
        w = d["s"]["windows"]
        assert abs(float(w[d["s"]["train_idx"]].mean())) < 0.1
        assert float(w[d["s"]["test_idx"]].mean()) > 10

    def test_windows_raw_preserves_pre_normalization_counts(self):
        """The Poisson reconstruction target must stay non-negative raw counts, and
        must not be clobbered if a caller already supplied it."""
        d = {"s": _session()}
        before = d["s"]["windows"].copy()
        normalize_session_windows(d, ["s"])
        assert np.allclose(d["s"]["windows_raw"], before)
        assert not np.allclose(d["s"]["windows"], before)

        d2 = {"s": _session()}
        sentinel = np.full_like(d2["s"]["windows"], 7.0)
        d2["s"]["windows_raw"] = sentinel.copy()
        normalize_session_windows(d2, ["s"])
        assert np.allclose(d2["s"]["windows_raw"], sentinel)


class TestForceSessionSpecific:
    def test_zeros_shared_and_makes_all_specific(self):
        """Cross-mouse neuron IDs are meaningless to tie together, so the pooled runs
        force every neuron session-specific. A stale shared block would silently mix
        unrelated neurons through one projection."""
        configs = {
            "s": {
                "n_total": 3,
                "all_ids": [11, 22, 33],
                "shared_indices": [0],
                "n_shared": 1,
            }
        }
        force_session_specific(configs)
        c = configs["s"]
        assert c["n_shared"] == 0 and c["shared_indices"] == []
        assert c["n_specific"] == 3 and c["specific_indices"] == [0, 1, 2]
        assert c["specific_ids"] == [11, 22, 33]
