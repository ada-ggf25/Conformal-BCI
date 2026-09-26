"""Tests for src/chain/pool.py.

Expected thresholds are order statistics worked by hand: at alpha = 0.1 a block of n scores has
the ceil((n + 1) * 0.9)-th smallest as its conformal quantile, and a region needs more than 9
windows for its own.
"""

from __future__ import annotations

import numpy as np

from src.chain.pool import BORROWED, GLOBAL, OWN, marginal_qhats, pooled_qhats


def _block(pairs):
    """A calibration block from (region, score) pairs."""
    return {
        "labels": np.array([r for r, _ in pairs]),
        "scores": np.array([s for _, s in pairs], dtype=float),
    }


# Own block: region 0 has 10 scores 0.1..1.0 (enough for its own), region 1 only 3.
OWN_BLOCK = _block(
    [(0, round(0.1 * k, 1)) for k in range(1, 11)] + [(1, 0.5), (1, 0.6), (1, 0.7)]
)
OLDER = _block([(1, 5.0)] * 20 + [(0, 9.0)] * 20)
NEWER = _block([(1, 2.0)] * 10 + [(0, 9.0)] * 20)


def test_borrowing_only_fills_starved_regions_from_the_most_recent_sessions():
    """B1: region 0 keeps its own quantile (past scores ignored); region 1 borrows from the
    newest session only (13 scores -> 13th smallest = 2.0, never the older 5.0); region 2,
    absent everywhere, takes the session's global threshold (13 own scores -> 13th = 1.0)."""
    q, lv = pooled_qhats("B1", OWN_BLOCK, [OLDER, NEWER], n_classes=3, alpha=0.1)
    assert q == {0: 1.0, 1: 2.0, 2: 1.0}
    assert lv == {0: OWN, 1: BORROWED, 2: GLOBAL}


def test_rule_a_never_borrows():
    """A starved region falls straight back to the session's global threshold."""
    q, lv = pooled_qhats("A", OWN_BLOCK, [OLDER, NEWER], n_classes=3, alpha=0.1)
    assert q[1] == 1.0 and lv[1] == GLOBAL


def test_standard_conformal_has_one_threshold_for_every_region():
    """The baseline's split conformal: the own block's quantile (13th of 13 scores = 1.0)."""
    q, lv = marginal_qhats(OWN_BLOCK, n_classes=3, alpha=0.1)
    assert q == {0: 1.0, 1: 1.0, 2: 1.0} and set(lv.values()) == {GLOBAL}
