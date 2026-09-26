"""Tests for src/chain/evaluate.py."""

from __future__ import annotations

import numpy as np

from src.chain.evaluate import evaluate_block


def test_coverage_size_error_and_level_per_window():
    """With sigma = 1 and every threshold 0.5, a set holds each region with p >= 0.5.

    Window 1: p = (0.6, 0.3, 0.1), truth 0 -> set {0}: covered, size 1/3, correct.
    Window 2: p = (0.5, 0.5, 0.0), truth 2 -> set {0, 1}: not covered, size 2/3, wrong.
    """
    test = {
        "probs": np.array([[0.6, 0.3, 0.1], [0.5, 0.5, 0.0]]),
        "sigma": np.ones(2),
        "labels": np.array([0, 2]),
    }
    ev = evaluate_block(
        test, dict.fromkeys(range(3), 0.5), {0: "own", 1: "own", 2: "global"}, 3
    )
    assert ev["covered"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(ev["size_over_k"], [1 / 3, 2 / 3])
    assert ev["error"].tolist() == [0.0, 1.0]
    assert ev["level"].tolist() == ["own", "global"]
