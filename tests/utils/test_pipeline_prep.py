"""Tests for src/utils/pipeline_prep.py.

The extraction functions ``assemble_session_data`` calls are patched, so each session's windows,
labels and task mask are chosen by hand and every gate is driven on purpose.

Accepted gaps: none.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.data.config import MIN_NEURONS_PER_SESSION, WINDOW_SIZE
from src.utils.pipeline_prep import (
    apply_window_region_labels,
    assemble_session_data,
    assert_chronological_disjoint,
    compute_class_counts,
    force_session_specific,
    gate_sessions_conformal,
    normalize_session_windows,
)


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


class TestAssembleSessionData:
    """Windows whose Linear trial has no water position have no region, so they must never
    reach the train, calibration or test blocks."""

    @staticmethod
    def _run(paradigm):
        n_windows, win = 30, 10
        n_bins = n_windows * win
        y = np.linspace(
            0.0, 2400.0, n_bins
        )  # a run toward water at 2500: several regions
        water = np.full(n_bins, 2500.0)
        water[5 * win : 6 * win] = np.nan  # window 5 belongs to a trial with no water
        config = {"n_total": 6}
        with (
            patch(
                "src.utils.pipeline_prep.build_hybrid_neuron_mapping",
                return_value=([], {"S": config}),
            ),
            patch(
                "src.utils.pipeline_prep.extract_neural_and_position_hybrid",
                return_value=(
                    np.zeros((n_bins, 6), "float32"),
                    np.zeros(n_bins),
                    y,
                    water,
                ),
            ),
            patch(
                "src.utils.pipeline_prep.window_task_mask",
                return_value=np.ones(n_windows, bool),
            ),
        ):
            session_data, *_ = assemble_session_data(
                None, ["S"], paradigm, train_frac=0.6, val_frac=0.3, test_frac=0.1
            )
        d = session_data["S"]
        return set(
            np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]]).tolist()
        )

    def test_linear_windows_without_water_are_dropped(self):
        """Window 5 is excluded from every block on the Linear maze."""
        used = self._run("More_Water")
        assert 5 not in used and len(used) == 29

    def test_other_mazes_ignore_the_water_position(self):
        """The Y maze does not use water, so the same window is kept."""
        assert 5 in self._run("Y_Maze")

    def test_normalized_train_block_has_unit_spread_per_neuron(self):
        """Each neuron is z-scored separately: per-neuron train mean 0 and std ~1 (the +1e-3
        guard only shrinks it)."""
        rng = np.random.default_rng(3)
        w = rng.normal(size=(40, 5, 3)) * np.array([1.0, 10.0, 100.0]) + np.array(
            [0.0, 5.0, -50.0]
        )
        d = {"s": {"windows": w.astype("float32"), "train_idx": np.arange(30)}}
        normalize_session_windows(d, ["s"])
        tr = d["s"]["windows"][:30]
        np.testing.assert_allclose(tr.mean(axis=(0, 1)), 0.0, atol=1e-4)
        np.testing.assert_allclose(tr.std(axis=(0, 1)), 1.0, atol=2e-3)


# --------------------------------------------------------------------------- assembly gates


def _assemble(sessions, *, min_windows=20, min_classes=2):
    """Run assemble_session_data over hand-built sessions.

    ``sessions`` maps a name to ``(n_total, n_windows, labels, mask_len, n_task)``: the neuron
    count, the extracted window count (0 = extraction failed), the per-window Y-maze region
    (via y positions), the task mask length, and how many leading windows are task windows.
    """
    ylevel = {0: 0.0, 1: 2000.0, 4: 3000.0}  # Y-maze: R0, R1, R4 at x = 0
    configs = {s: {"n_total": v[0]} for s, v in sessions.items()}

    def extract(_db, s, _cfg, _bw):
        """Extracted bins for session s, or Nones when it has no windows."""
        _n, n_win, labels, *_ = sessions[s]
        if n_win == 0:
            return None, None, None, None
        y = np.repeat([ylevel[r] for r in labels], WINDOW_SIZE)
        n_bins = n_win * WINDOW_SIZE
        return (
            np.zeros((n_bins, 2), "float32"),
            np.zeros(n_bins),
            y,
            np.full(n_bins, np.nan),
        )

    def mask(_db, s, _w, _bw):
        """Task mask for s: its first n_task windows are task windows."""
        *_, mask_len, n_task = sessions[s]
        m = np.zeros(mask_len, bool)
        m[:n_task] = True
        return m

    with (
        patch(
            "src.utils.pipeline_prep.build_hybrid_neuron_mapping",
            return_value=(["u1", "u2"], configs),
        ),
        patch(
            "src.utils.pipeline_prep.extract_neural_and_position_hybrid",
            side_effect=extract,
        ),
        patch("src.utils.pipeline_prep.window_task_mask", side_effect=mask),
    ):
        return assemble_session_data(
            None,
            list(sessions),
            "Y_Maze",
            min_windows=min_windows,
            min_classes=min_classes,
        )


def test_every_session_gate():
    """Each gate drops exactly the session built to fail it; the good one survives."""
    two = [0, 1] * 15  # 30 windows, two regions
    n = MIN_NEURONS_PER_SESSION
    data, valid, configs, shared, n_shared = _assemble(
        {
            "good": (n, 30, two, 30, 30),
            "few_neurons": (n - 1, 30, two, 30, 30),
            "no_trials": (n, 0, [], 0, 0),
            "misaligned_mask": (n, 30, two, 29, 29),
            "few_task_windows": (n, 30, two, 30, 19),
            "one_region": (n, 30, [1] * 30, 30, 30),
        }
    )
    assert valid == ["good"] and list(data) == ["good"]
    assert set(configs) == {
        "good", "few_neurons", "no_trials", "misaligned_mask", "few_task_windows", "one_region",
    }  # fmt: skip
    assert shared == ["u1", "u2"] and n_shared == 2


def test_region_gate_counts_task_windows_only():
    """A second region seen only outside the task windows does not rescue a session."""
    labels = [1] * 25 + [4] * 5
    _, valid, *_ = _assemble({"s": (9, 30, labels, 30, 25)})
    assert valid == []


def test_assembled_session_keeps_positions_and_the_default_split():
    """A kept session stores its positions and the 80/10/10 chronological task split."""
    data, *_ = _assemble({"s": (9, 30, [0, 1] * 15, 30, 30)})
    d = data["s"]
    assert (len(d["train_idx"]), len(d["val_idx"]), len(d["test_idx"])) == (24, 3, 3)
    assert d["regions"].tolist() == [0, 1] * 15
    assert len(d["y_positions"]) == 30 * WINDOW_SIZE
    assert set(d) >= {"windows", "x_positions", "water_positions"}


# --------------------------------------------------------------------------- relabeling


def _relabel_session():
    """Three Y-maze windows of 4 bins: pure R0, R1-majority ending in R0, pure R1."""
    y = np.array([0.0] * 4 + [1300.0, 1300.0, 1300.0, 0.0] + [1300.0] * 4)
    return {
        "x_positions": np.zeros(12),
        "y_positions": y,
        "regions": np.array([0, 0, 1]),  # last-bin labels
        "train_idx": np.array([0, 1]),
        "val_idx": np.array([], int),
        "test_idx": np.array([2]),
    }


class TestApplyWindowRegionLabels:
    def test_mode_relabels_and_drops_straddlers(self):
        """The straddling window 1 is relabeled R1 by its mode and removed from every block."""
        d = {"s": _relabel_session()}
        n_straddled, n_total = apply_window_region_labels(
            d, ["s"], "Y_Maze", reduce="mode", drop_straddle=True, window_size=4
        )
        assert (n_straddled, n_total) == (1, 3)
        assert d["s"]["regions"].tolist() == [0, 1, 1]
        assert d["s"]["straddle"].tolist() == [False, True, False]
        assert d["s"]["train_idx"].tolist() == [0]
        assert d["s"]["test_idx"].tolist() == [2]

    def test_last_keeps_labels_and_blocks_when_not_dropping(self):
        """reduce="last" leaves the extraction's labels alone; without drop the blocks stay."""
        d = {"s": _relabel_session()}
        apply_window_region_labels(
            d, ["s"], "Y_Maze", reduce="last", drop_straddle=False, window_size=4
        )
        assert d["s"]["regions"].tolist() == [0, 0, 1]
        assert d["s"]["train_idx"].tolist() == [0, 1]
        assert d["s"]["straddle"].tolist() == [False, True, False]

    def test_too_few_position_bins_skips_the_session_with_a_warning(self):
        """If positions cover fewer windows than exist, the session is left untouched."""
        d = {"s": _relabel_session()}
        d["s"]["y_positions"] = d["s"]["y_positions"][:8]
        d["s"]["x_positions"] = d["s"]["x_positions"][:8]
        logs: list[str] = []
        out = apply_window_region_labels(
            d, ["s"], "Y_Maze", reduce="mode", drop_straddle=True, window_size=4,
            log_fn=logs.append,
        )  # fmt: skip
        assert out == (0, 0)
        assert "straddle" not in d["s"] and len(logs) == 1 and "warn s" in logs[0]
        assert apply_window_region_labels(
            d, ["s"], "Y_Maze", reduce="mode", drop_straddle=True, window_size=4
        ) == (0, 0)  # the log sink is optional


# --------------------------------------------------------------------------- conformal gates


def test_gate_drops_sessions_with_small_val_or_test_blocks():
    """Sessions below either floor are removed from both the list and the data, in place."""

    def s(n_val, n_test):
        """A session with the given block sizes."""
        return {"val_idx": np.arange(n_val), "test_idx": np.arange(n_test)}

    data = {"ok": s(10, 5), "small_val": s(9, 5), "small_test": s(10, 4)}
    valid = ["ok", "small_val", "small_test"]
    dropped = gate_sessions_conformal(data, valid, min_val=10, min_test=5)
    assert dropped == [("small_val", 9, 5), ("small_test", 10, 4)]
    assert valid == ["ok"] and list(data) == ["ok"]


class TestAssertChronologicalDisjoint:
    def test_ordered_blocks_pass(self):
        """Train before val before test passes, regardless of the index order stored."""
        d = {"s": {"train_idx": np.array([2, 0, 1]), "val_idx": np.array([3, 4]),
                   "test_idx": np.array([6, 5])}}  # fmt: skip
        assert_chronological_disjoint(d, ["s"])

    @pytest.mark.parametrize(
        ("tr", "va", "te"),
        [([0, 1, 5], [3, 4], [6]), ([0, 1], [2, 4], [4, 5]), ([0, 3], [2], [5])],
    )
    def test_interleaved_or_overlapping_blocks_fail(self, tr, va, te):
        """Any train window after a val window, or shared windows, breaks the split."""
        d = {"s": {"train_idx": np.array(tr), "val_idx": np.array(va),
                   "test_idx": np.array(te)}}  # fmt: skip
        with pytest.raises(AssertionError, match="not chronological"):
            assert_chronological_disjoint(d, ["s"])


def test_class_counts_pool_train_rows_of_the_listed_sessions():
    """Only train rows of listed, present sessions count; val/test rows never do."""
    data = {
        "a": {
            "regions": np.array([0, 1, 1, 2, 2, 2]),
            "train_idx": np.array([0, 1, 3]),
        },
        "b": {"regions": np.array([1, 1, 5]), "train_idx": np.array([0, 2])},
        "c": {"regions": np.array([7]), "train_idx": np.array([], int)},
        "unlisted": {"regions": np.array([9]), "train_idx": np.array([0])},
    }
    assert compute_class_counts(data, ["a", "b", "c", "missing"]) == {
        0: 1,
        1: 2,
        2: 1,
        5: 1,
    }
    assert compute_class_counts(data, ["c"]) == {}
