"""Tests for src/conformal/evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.calibrate import compute_quantile
from src.conformal.evaluate import (
    build_offsets,
    class_conditional_q_scaled,
    pool_indices,
)


def _conformal_arrays(n=40, k=3, seed=0):
    """Return (softmax, random labels) for n windows over k classes."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, k))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return probs, rng.integers(0, k, n)


class TestPooledIndexing:
    def test_pool_indices_sorts_and_shifts(self):
        """The conformal layer reads probs[val_i]/labels[val_i] through these indices,
        so a dropped sort or a wrong offset calibrates q_hat on the wrong windows.
        """
        sd = {  # sessions of 5 and 4 windows, blocks deliberately unsorted
            "a": {"regions": np.zeros(5), "train_idx": np.array([3, 0, 1])},
            "b": {"regions": np.zeros(4), "train_idx": np.array([2, 1])},
        }
        off = build_offsets(sd, ["a", "b"])
        # a: sorted[0,1,3] + 0 , b: sorted[1,2] + 5
        assert list(pool_indices(sd, ["a", "b"], "train_idx", off)) == [0, 1, 3, 6, 7]


class TestClassConditionalQScaled:
    """The score is (1 - p_true)/sigma over the val block only. Misindexing p_true,
    forgetting to subset labels alongside val_i, or dropping sigma are all silent.
    The dict still has one finite threshold per class."""

    def test_scores_are_true_class_residuals_over_the_val_block_only(self):
        """Each threshold is compute_quantile over (1 - p_true)/sigma on that class's val rows."""
        probs, labels = _conformal_arrays(n=60, k=2)
        sigma = np.linspace(0.5, 2.0, len(labels))
        val_i = np.arange(20, 60)  # a strict subset: rows 0-19 must not be read
        q = class_conditional_q_scaled(probs, labels, sigma, val_i, 2, 0.1)
        # Rebuild the expected threshold per class straight from the definition.
        for c in (0, 1):
            rows = [i for i in val_i if labels[i] == c]
            scores = np.array([(1.0 - probs[i, labels[i]]) / sigma[i] for i in rows])
            assert q[c] == pytest.approx(compute_quantile(scores, 0.1))

    def test_rows_outside_val_i_are_ignored(self):
        """Corrupting rows outside val_i leaves every threshold unchanged."""
        probs, labels = _conformal_arrays(n=60, k=2)
        sigma, val_i = np.ones(len(labels)), np.arange(20, 60)
        before = class_conditional_q_scaled(probs, labels, sigma, val_i, 2, 0.1)
        probs[:20] = 0.0  # corrupt only excluded rows, thresholds must not move
        after = class_conditional_q_scaled(probs, labels, sigma, val_i, 2, 0.1)
        assert before == after
