"""Tests for src/conformal/predict.py."""

from __future__ import annotations

import numpy as np

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
