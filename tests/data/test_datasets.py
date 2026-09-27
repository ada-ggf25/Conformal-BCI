"""Tests for src/data/datasets.py.

These datasets decide which windows each training step sees and with which label and session,
so the tests pin the row selection and the pairing, both worked out by hand.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.datasets import (
    HybridClassificationDataset,
    LatentSequenceDataset,
    hybrid_collate_fn,
)


def _session(offset: float, n: int = 6, with_raw: bool = True) -> dict:
    """n windows of shape (2, 3) whose values encode their row: window i is filled with offset+i."""
    w = (offset + np.arange(n, dtype=np.float32))[:, None, None] * np.ones(
        (1, 2, 3), np.float32
    )
    d = {
        "windows": w,
        "regions": np.arange(n) % 3,
        "train_idx": np.array([0, 2, 3]),
        "val_idx": np.array([4, 5]),
    }
    if with_raw:
        d["windows_raw"] = w + 1000.0
    return d


class TestHybridClassificationDataset:
    def test_only_the_requested_split_rows_are_used(self):
        """Every sample is a row of the chosen split, paired with that row's own region."""
        ds = HybridClassificationDataset({"A": _session(0.0)}, index_key="val_idx")
        assert len(ds) == 2
        got = [(float(ds[i]["window"][0, 0]), ds[i]["region"]) for i in range(2)]
        assert got == [(4.0, 4 % 3), (5.0, 5 % 3)]

    def test_samples_carry_their_session_and_raw_counts(self):
        """Each sample keeps its session name and the same row of the raw-count array."""
        ds = HybridClassificationDataset({"A": _session(0.0), "B": _session(100.0)})
        assert len(ds) == 6
        names = [ds[i]["session_name"] for i in range(6)]
        assert names == ["A"] * 3 + ["B"] * 3
        s = ds[4]  # B's train row 2
        assert float(s["window"][0, 0]) == 102.0
        assert float(s["window_raw"][0, 0]) == 1102.0
        assert s["window"].dtype == torch.float32

    def test_raw_counts_fall_back_to_the_windows(self):
        """Without windows_raw, the reconstruction target is the window itself."""
        ds = HybridClassificationDataset({"A": _session(0.0, with_raw=False)})
        assert torch.equal(ds[1]["window"], ds[1]["window_raw"])


def test_collate_groups_samples_by_session_in_arrival_order():
    """A mixed batch splits into one stacked tensor per session, rows in arrival order."""

    def sample(name, v, region):
        """One dataset sample whose window is filled with v."""
        return {
            "window": torch.full((2, 3), v),
            "window_raw": torch.full((2, 3), v + 0.5),
            "region": region,
            "session_name": name,
        }

    batch = [sample("A", 1.0, 0), sample("B", 2.0, 1), sample("A", 3.0, 2)]
    out = hybrid_collate_fn(batch)
    assert list(out) == ["A", "B"]
    assert out["A"]["windows"][:, 0, 0].tolist() == [1.0, 3.0]
    assert out["A"]["windows_raw"][:, 0, 0].tolist() == [1.5, 3.5]
    assert out["A"]["regions"].tolist() == [0, 2]
    assert out["A"]["regions"].dtype == torch.long
    assert out["B"]["windows"].shape == (1, 2, 3)


class TestLatentSequenceDataset:
    @staticmethod
    def _data():
        """Two sessions of scalar latents 0..4 and 10..14 (latent_dim = 1)."""
        return {
            "S1": {
                "step1_all_latents": np.arange(5, dtype=np.float32)[:, None],
                "regions": np.array([0, 1, 2, 0, 1]),
                "train_idx": np.array([0, 3, 4]),
            },
            "S2": {
                "step1_all_latents": np.arange(10, 15, dtype=np.float32)[:, None],
                "regions": np.array([2, 2, 1, 1, 0]),
                "train_idx": np.array([1]),
            },
        }

    def test_sequences_end_at_their_window_and_are_front_padded(self):
        """Row i's sequence is latents i-2..i; before the start the first latent repeats."""
        ds = LatentSequenceDataset(self._data(), ["S1", "S2"], seq_len=3)
        assert len(ds) == 4
        seqs = [ds[i][0][:, 0].tolist() for i in range(4)]
        assert seqs == [[0, 0, 0], [1, 2, 3], [2, 3, 4], [10, 10, 11]]

    def test_labels_and_session_ids_pair_with_their_rows(self):
        """Each sequence carries its own window's region and its session's position."""
        ds = LatentSequenceDataset(self._data(), ["S2", "S1"], seq_len=3)
        pairs = [(int(ds[i][1]), int(ds[i][2])) for i in range(4)]
        # S2 listed first -> id 0; its train row 1 has region 2. S1 rows 0, 3, 4 -> id 1.
        assert pairs == [(2, 0), (0, 1), (0, 1), (1, 1)]

    def test_context_comes_from_all_windows_not_only_the_split(self):
        """The context before a val row includes train rows: only the endpoint is filtered."""
        data = self._data()
        data["S1"]["val_idx"] = np.array([2])
        ds = LatentSequenceDataset(data, ["S1"], seq_len=3, index_key="val_idx")
        assert ds[0][0][:, 0].tolist() == [0, 1, 2]
