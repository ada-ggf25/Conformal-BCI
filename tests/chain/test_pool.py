"""Tests for src/chain/pool.py.

Expected thresholds are order statistics worked by hand: at alpha = 0.1 a block of n scores has
the ceil((n + 1) * 0.9)-th smallest as its conformal quantile, and a region needs more than 9
windows for its own.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chain.pool import (
    BORROWED,
    GLOBAL,
    OWN,
    borrow_depth,
    marginal_qhats,
    pooled_qhats,
    region_floor,
)
from src.conformal.calibrate import compute_quantile


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


@pytest.mark.parametrize(
    ("rule", "depth"), [("A", 0), ("B1", 1), ("B3", 3), ("B12", 12)]
)
def test_borrow_depth(rule, depth):
    """A never borrows; B<M> borrows from the last M sessions."""
    assert borrow_depth(rule) == depth


@pytest.mark.parametrize("rule", ["B", "Bx", "C3", "b3", ""])
def test_unknown_rules_are_refused(rule):
    """A typo in the rule name must not silently mean "no borrowing"."""
    with pytest.raises(ValueError, match="unknown pool rule"):
        borrow_depth(rule)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.25, 0.3])
def test_the_floor_is_the_smallest_block_with_a_finite_quantile(alpha):
    """ceil((n + 1)(1 - alpha)) <= n first holds at n = ceil(1/alpha) - 1: a block that size
    has a finite quantile (its largest score), one window fewer has none."""
    n = region_floor(alpha)
    scores = np.arange(n, dtype=float)
    assert compute_quantile(scores, alpha) == scores.max()
    assert compute_quantile(scores[:-1], alpha) == np.inf


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.25, 0.3])
def test_a_region_above_the_floor_always_has_a_finite_quantile(alpha):
    """The floor exists so that an "own" or "borrowed" threshold is never +inf: one window
    more than the floor always satisfies ceil((n + 1)(1 - alpha)) <= n."""
    n = region_floor(alpha) + 1
    assert np.isfinite(compute_quantile(np.arange(n, dtype=float), alpha))


def test_borrowing_uses_only_the_last_m_sessions():
    """B2 over five past sessions reads the last two only: region 1 pools the own 3 scores
    with 4 + 4 recent ones (11 scores, 11th smallest = the max, 3.0); the older sessions'
    100.0 never enters."""
    old = _block([(1, 100.0)] * 20)
    recent = _block([(1, 3.0)] * 4)
    q, lv = pooled_qhats("B2", OWN_BLOCK, [old, old, old, recent, recent], 3, 0.1)
    assert q[1] == 3.0 and lv[1] == BORROWED


def test_a_region_at_exactly_the_floor_is_still_starved():
    """Nine own windows at alpha 0.1 are not enough for an own threshold."""
    own = _block([(0, float(k)) for k in range(9)] + [(1, 50.0)] * 10)
    q, lv = pooled_qhats("A", own, [], n_classes=2, alpha=0.1)
    assert lv == {0: GLOBAL, 1: OWN}
    # global over the 19 own scores: ceil(20 * 0.9) = 18th smallest = 50.0
    assert q[0] == 50.0


def test_global_fallback_ignores_the_borrowed_scores():
    """A region starved even after pooling falls back to the own block's quantile, not to
    one over the pooled past."""
    q, lv = pooled_qhats("B3", OWN_BLOCK, [_block([(0, 99.0)] * 20)], 3, 0.1)
    assert lv[2] == GLOBAL and q[2] == 1.0  # own 13 scores -> 13th smallest


def test_marginal_with_a_tiny_block_admits_everything():
    """Too few calibration windows for any finite threshold: every q_hat is +inf."""
    q, _ = marginal_qhats(_block([(0, 0.1)] * 3), n_classes=2, alpha=0.1)
    assert q == {0: np.inf, 1: np.inf}


def test_floor_scales_with_alpha():
    """At alpha 0.2 a region needs more than ceil(1/0.2) - 1 = 4 windows: 4 is starved, 5 is own."""
    own = _block([(0, 1.0)] * 4 + [(1, 1.0)] * 5)
    _, lv = pooled_qhats("A", own, [], n_classes=2, alpha=0.2)
    assert lv == {0: GLOBAL, 1: OWN}
    assert region_floor(0.2) == 4 and region_floor(0.05) == 19
