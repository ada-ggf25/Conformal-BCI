"""Chain order, the warm-up split, and the first-session flags."""

from __future__ import annotations

import numpy as np

from ..utils.pipeline_prep import normalize_session_windows


def chain_order(sessions: list[str]) -> list[str]:
    """Calendar order across mice: by recording date (``M<id>_<YYMMDD>``), ties by mouse."""
    return sorted(sessions, key=lambda s: (s.split("_")[1], s.split("_")[0]))


def resplit_warmup(d: dict, train_frac: float = 0.7) -> None:
    """Fold a warm-up session's task windows into a train / calibration split, no test block.

    Warm-up sessions are never tested. Their task windows (every index of the original split)
    are re-cut chronologically: the first ``train_frac`` train, the rest calibration.
    """
    task = np.sort(np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]]))
    cut = round(train_frac * len(task))
    d["train_idx"], d["val_idx"], d["test_idx"] = task[:cut], task[cut:], task[:0]


def first_session_flags(chain: list[str], mouse: dict[str, str]) -> list[bool]:
    """Per chain position: is this the first session of its mouse in the chain?"""
    seen: set[str] = set()
    flags = []
    for s in chain:
        flags.append(mouse[s] not in seen)
        seen.add(mouse[s])
    return flags


def prepare_chain(
    session_data: dict,
    sessions: list[str],
    n_warmup: int,
    warmup_train_frac: float = 0.7,
) -> list[str]:
    """Order the sessions, re-split the warm-up ones, then z-score every session.

    The warm-up re-split must come before normalization, because each session's neurons are
    z-scored with its own training-block statistics.

    Returns:
        The chain: session names in calendar order.
    """
    chain = chain_order(sessions)
    for s in chain[:n_warmup]:
        resplit_warmup(session_data[s], warmup_train_frac)
    normalize_session_windows(session_data, chain)
    return chain
