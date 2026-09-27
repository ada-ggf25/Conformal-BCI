"""Per-region conformal thresholds for a session, and which data each one came from.

A calibration block is a dict with ``"scores"`` (the true-class nonconformity scores) and
``"labels"`` (the true regions). Rule ``A`` uses the session's own block only. Rule ``B<M>``
uses the own block for every region that has enough windows in it; a starved region instead
takes its threshold from the own block pooled with the last ``M`` sessions' calibration blocks;
a region still starved falls back to the session's global threshold.
"""

from __future__ import annotations

import numpy as np

from ..conformal.calibrate import compute_quantile

OWN, BORROWED, GLOBAL = "own", "borrowed", "global"


def region_floor(alpha: float) -> int:
    """A region needs more than this many calibration windows for its own quantile.

    ``ceil(1/alpha) - 1`` is the smallest block with a finite conformal quantile, and there the
    quantile is the block's largest score. Requiring more (at least ``ceil(1/alpha)`` windows, the
    usual rule of thumb) sends a region that small to the pooled or global threshold instead.
    """
    return int(np.ceil(1.0 / alpha)) - 1


def borrow_depth(rule: str) -> int:
    """How many past sessions a rule may borrow from: ``"A"`` -> 0, ``"B3"`` -> 3."""
    if rule == "A":
        return 0
    if rule.startswith("B") and rule[1:].isdigit():
        return int(rule[1:])
    raise ValueError(f"unknown pool rule {rule!r}")


def pooled_qhats(
    rule: str,
    own: dict[str, np.ndarray],
    past: list[dict[str, np.ndarray]],
    n_classes: int,
    alpha: float,
) -> tuple[dict[int, float], dict[int, str]]:
    """Per-region thresholds for one session and the level each came from.

    Args:
        rule:      ``"A"`` or ``"B<M>"``.
        own:       this session's calibration block.
        past:      earlier sessions' calibration blocks, oldest first, already scored with this
                   session's model; the last ``M`` are used.
        n_classes: number of regions.
        alpha:     miscoverage level.

    Returns:
        ``(qhats, levels)``: ``{region: q_hat}`` and ``{region: "own" | "borrowed" | "global"}``.
    """
    m = borrow_depth(rule)
    floor = region_floor(alpha)
    sel = past[-m:] if m else []
    pool_s = np.concatenate([own["scores"], *[p["scores"] for p in sel]])
    pool_y = np.concatenate([own["labels"], *[p["labels"] for p in sel]])
    own_global = compute_quantile(own["scores"], alpha)
    qhats: dict[int, float] = {}
    levels: dict[int, str] = {}
    for c in range(n_classes):
        own_c = own["scores"][own["labels"] == c]
        pool_c = pool_s[pool_y == c]
        if len(own_c) > floor:
            qhats[c], levels[c] = compute_quantile(own_c, alpha), OWN
        elif len(pool_c) > floor:
            qhats[c], levels[c] = compute_quantile(pool_c, alpha), BORROWED
        else:
            qhats[c], levels[c] = own_global, GLOBAL
    return qhats, levels


def marginal_qhats(
    own: dict[str, np.ndarray], n_classes: int, alpha: float
) -> tuple[dict[int, float], dict[int, str]]:
    """Standard split conformal: one threshold for every region, from the own block."""
    q = compute_quantile(own["scores"], alpha)
    return dict.fromkeys(range(n_classes), q), dict.fromkeys(range(n_classes), GLOBAL)
