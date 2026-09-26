"""Tests for src/chain/order.py.

Accepted gaps: ``prepare_chain`` is orchestration (order, re-split, then the train-only
normalization already tested in tests/utils/test_pipeline_prep.py).
"""

from __future__ import annotations

import numpy as np

from src.chain.order import chain_order, first_session_flags, resplit_warmup


def test_chain_is_calendar_order_with_ties_broken_by_mouse():
    """Sessions of different mice interleave by date; the same date sorts by mouse id."""
    sessions = ["M2_240110", "M1_240111", "M1_240110", "M3_240109"]
    assert chain_order(sessions) == ["M3_240109", "M1_240110", "M2_240110", "M1_240111"]


def test_warmup_resplit_is_chronological_70_30_with_no_test_block():
    """A warm-up session is never tested: all its task windows become train then calibration."""
    d = {
        "train_idx": np.arange(60),
        "val_idx": np.arange(60, 90),
        "test_idx": np.arange(90, 100),
    }
    resplit_warmup(d, 0.7)
    assert d["train_idx"].tolist() == list(range(70))
    assert d["val_idx"].tolist() == list(range(70, 100))
    assert len(d["test_idx"]) == 0


def test_first_session_of_each_mouse_is_flagged_once():
    """A mouse is new only at its first chain position."""
    chain = ["M1_1", "M2_1", "M1_2", "M3_1", "M2_2"]
    mouse = {s: s.split("_")[0] for s in chain}
    assert first_session_flags(chain, mouse) == [True, True, False, True, False]
