"""Tests for src/utils/splits.py.

Accepted gaps: none.
"""

import numpy as np
import pytest

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


def test_fractions_must_sum_to_one():
    """A split that does not partition the task windows is refused."""
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        chronological_task_split(np.ones(10, bool), 0.7, 0.2, 0.2)


def test_custom_fractions_with_non_task_windows_interleaved():
    """60/30/10 over the 10 task windows among 15: blocks of 6, 3 and 1 task windows, in order."""
    mask = np.ones(15, bool)
    mask[[1, 4, 7, 10, 13]] = False
    task = np.flatnonzero(mask).tolist()
    tr, va, te = chronological_task_split(mask, 0.6, 0.3, 0.1)
    assert (
        tr.tolist() == task[:6] and va.tolist() == task[6:9] and te.tolist() == task[9:]
    )


def test_no_task_windows_gives_three_empty_blocks():
    """A session with no task windows has nothing to split."""
    assert [len(b) for b in chronological_task_split(np.zeros(5, bool))] == [0, 0, 0]
