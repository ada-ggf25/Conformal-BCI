"""Tests for src/chain/order.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chain.order import (
    chain_order,
    first_session_flags,
    prepare_chain,
    resplit_warmup,
)


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


def test_warmup_resplit_keeps_only_task_windows():
    """Non-task windows (never in any block) stay out after the re-split."""
    d = {
        "train_idx": np.array([0, 2, 4]),
        "val_idx": np.array([6]),
        "test_idx": np.array([8]),
    }
    resplit_warmup(d, 0.6)
    assert d["train_idx"].tolist() == [0, 2, 4] and d["val_idx"].tolist() == [6, 8]


def test_prepare_chain_resplits_warmup_before_normalizing():
    """The warm-up session's z-scoring uses its new 70% train block (six 0s and a 1), not the
    original 6-window one; a tested session keeps its split and is z-scored on it."""

    def session(values, tr, va, te):
        """One neuron, T = 1, the given per-window values and blocks."""
        return {
            "windows": np.array(values, dtype="float32")[:, None, None],
            "train_idx": np.array(tr), "val_idx": np.array(va), "test_idx": np.array(te),
        }  # fmt: skip

    data = {
        "M1_240102": session(
            [0.0] * 6 + [1.0, 10.0, 10.0, 10.0], range(6), [6, 7, 8], [9]
        ),
        "M1_240101": session(
            [0.0] * 6 + [1.0, 10.0, 10.0, 10.0], range(6), [6, 7, 8], [9]
        ),
    }
    chain = prepare_chain(data, ["M1_240102", "M1_240101"], n_warmup=1)
    assert chain == ["M1_240101", "M1_240102"]
    warm, tested = data["M1_240101"], data["M1_240102"]
    assert warm["train_idx"].tolist() == list(range(7)) and len(warm["test_idx"]) == 0
    assert tested["train_idx"].tolist() == list(range(6))
    assert tested["test_idx"].tolist() == [9]
    # warm-up train block (0 x6, 1) has mean 1/7: its zeros sit below 0, the 10s far above
    w = warm["windows"][:, 0, 0]
    assert w[:6].max() < 0 < w[6] < w[7]
    # the tested session's train block is all zeros: its std is only the 1e-3 guard
    assert tested["windows"][:6].max() == 0.0
    assert tested["windows"][9, 0, 0] == pytest.approx(10_000.0, rel=1e-4)
    assert "windows_raw" in warm and warm["windows_raw"][9, 0, 0] == 10.0


def test_chain_order_does_not_depend_on_the_input_order():
    """Any permutation of the same sessions gives the same chain."""
    sessions = ["M2_240110", "M1_240111", "M1_240110", "M3_240109", "M10_240110"]
    assert chain_order(sessions) == chain_order(sessions[::-1])
    assert (
        chain_order(sessions)[0] == "M3_240109"
        and chain_order(sessions)[-1] == "M1_240111"
    )


def test_a_single_mouse_is_new_only_once():
    """Every session after a mouse's first is a returning one."""
    chain = ["M1_1", "M1_2", "M1_3"]
    assert first_session_flags(chain, dict.fromkeys(chain, "M1")) == [
        True,
        False,
        False,
    ]
