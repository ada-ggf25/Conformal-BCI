"""Tests for src/utils/splits.py."""

import numpy as np

from src.utils.splits import chronological_task_split


def test_all_task_blocks_are_contiguous_and_chronological():
    """The default 80/10/10 split yields three contiguous, time-ordered, exhaustive blocks."""
    tr, va, te = chronological_task_split(np.ones(100, dtype=bool))  # 80/10/10
    assert len(tr) == 80 and len(va) == 10 and len(te) == 10
    assert tr.tolist() == list(range(0, 80))
    assert va.tolist() == list(range(80, 90))
    assert te.tolist() == list(range(90, 100))


def test_indices_reference_original_array_not_task_subset():
    """Returning positions within the task subset instead of the original array
    silently shifts every window's identity (same lengths, no error)."""
    mask = np.zeros(20, dtype=bool)
    mask[1::2] = True  # task windows at odd positions only
    all_idx = np.concatenate(chronological_task_split(mask))
    assert set(all_idx.tolist()).issubset(set(range(1, 20, 2)))
    assert all_idx.tolist() == sorted(all_idx.tolist())
    assert len(all_idx) == 10
