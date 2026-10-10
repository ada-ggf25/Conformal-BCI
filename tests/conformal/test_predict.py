"""Tests for src/conformal/predict.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal.predict import class_conditional_prediction_sets

N_CLASSES = 6


def _probs(n, seed=0):
    """Return n rows of sampled class posteriors."""
    return np.random.RandomState(seed).dirichlet(np.ones(N_CLASSES), size=n)


def test_scale_one_reduces_to_the_threshold():
    """With sigma == 1 the learned layer must collapse to standard split conformal:
    class c is in the set if (1 - p_c) <= q_c."""
    p = _probs(50)
    qhats = {c: 0.5 + 0.05 * c for c in range(N_CLASSES)}
    got = class_conditional_prediction_sets(p, np.ones(len(p)), qhats)
    want = [
        [c for c in range(N_CLASSES) if (1.0 - p[i, c]) <= qhats[c]]
        for i in range(len(p))
    ]
    assert got == want


def test_a_larger_scale_admits_more_classes():
    """The scaled score (1 - p) / sigma shrinks as sigma grows: at sigma 2 a class with
    1 - p = 0.8 clears q = 0.5 (0.4 <= 0.5), at sigma 1 it does not."""
    p = np.array([[0.2, 0.8], [0.2, 0.8]])
    sets = class_conditional_prediction_sets(p, np.array([1.0, 2.0]), {0: 0.5, 1: 0.5})
    assert sets == [[1], [0, 1]]


def test_each_class_uses_its_own_threshold():
    """Two classes with the same probability can differ in membership through their q_hat."""
    p = np.array([[0.5, 0.5]])
    assert class_conditional_prediction_sets(p, np.ones(1), {0: 0.4, 1: 0.5}) == [[1]]


def test_infinite_threshold_admits_the_class_and_empty_sets_are_allowed():
    """q_hat = +inf (too little calibration data) always admits; a set may also be empty."""
    p = np.array([[0.1, 0.9], [0.5, 0.5]])
    sets = class_conditional_prediction_sets(p, np.ones(2), {0: np.inf, 1: 0.0})
    assert sets == [[0], [0]]
    assert class_conditional_prediction_sets(p, np.ones(2), {0: 0.0, 1: 0.0}) == [
        [],
        [],
    ]


def test_scale_and_probabilities_must_align():
    """A scale vector of the wrong length is an error, not a broadcast."""
    with pytest.raises(ValueError, match="must align"):
        class_conditional_prediction_sets(
            _probs(3), np.ones(2), dict.fromkeys(range(6), 1.0)
        )


def test_every_class_needs_a_threshold():
    """A missing region threshold is an error rather than a silent exclusion."""
    with pytest.raises(KeyError):
        class_conditional_prediction_sets(_probs(2), np.ones(2), {0: 1.0})
