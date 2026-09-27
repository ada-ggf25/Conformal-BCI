"""Session data assembly and class-count utilities.

``assemble_session_data`` builds the per-session data dicts every later stage reads and
modifies in place: ``windows`` and ``regions``, the ``train_idx`` / ``val_idx`` /
``test_idx`` split, and the per-bin positions. The other helpers transform or check those
dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymongo.database import Database

from ..data.config import BIN_WIDTH_SEC, MIN_NEURONS_PER_SESSION, WINDOW_SIZE
from ..data.hybrid_io import (
    build_hybrid_neuron_mapping,
    create_windows_with_regions,
    extract_neural_and_position_hybrid,
    window_region_labels,
    window_task_mask,
)
from ..data.regions import uses_water
from .splits import chronological_task_split


def assemble_session_data(
    db: Database,
    sessions: list[str],
    paradigm: str,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    min_windows: int = 20,
    min_classes: int = 2,
) -> tuple[dict, list[str], dict, list[str], int]:
    """Extract, window, label, and split neural data for each session.

    A session is dropped when it has fewer than ``MIN_NEURONS_PER_SESSION`` neurons, no
    usable trial, a task mask that does not line up with its windows, fewer than
    ``min_windows`` task windows, or fewer than ``min_classes`` regions among them.

    Args:
        db:           pymongo Database
        sessions:     ordered list of session names to process
        paradigm:     maze type string (e.g. 'Y_Maze', 'Nested', 'More_Water'). For the
                      Linear maze, windows of trials without a usable water position are
                      kept out of the task windows (their regions are undefined).
        train_frac:   fraction for model training (default 0.80)
        val_frac:     fraction for early-stopping validation / q̂ (default 0.10)
        test_frac:    fraction for coverage evaluation (default 0.10)
        min_windows:  sessions with fewer task windows are dropped (default 20)
        min_classes:  sessions with fewer unique region classes are dropped (default 2)

    Returns:
        ``(session_data, valid_sessions, session_configs, shared_neuron_ids, n_shared)``.
        ``session_data`` maps each kept session to a dict with ``windows``, ``regions``,
        ``train_idx``, ``val_idx``, ``test_idx``, ``x_positions``, ``y_positions`` and
        ``water_positions``. ``valid_sessions`` lists the kept sessions in input order.
        ``session_configs`` has the neuron config of every input session (from
        ``build_hybrid_neuron_mapping``). ``shared_neuron_ids`` is the sorted list of
        neuron IDs present in all sessions, and ``n_shared`` its length.
    """
    shared_neuron_ids, session_configs = build_hybrid_neuron_mapping(db, sessions)
    n_shared = len(shared_neuron_ids)

    session_data: dict = {}
    valid_sessions: list[str] = []

    for session_name in sessions:
        config = session_configs[session_name]

        if config["n_total"] < MIN_NEURONS_PER_SESSION:
            continue

        neural_data, x_positions, y_positions, water = (
            extract_neural_and_position_hybrid(db, session_name, config, BIN_WIDTH_SEC)
        )
        if neural_data is None:
            continue

        windows, regions = create_windows_with_regions(
            neural_data, x_positions, y_positions, WINDOW_SIZE, paradigm, water
        )

        # Chronological within-session split over task (W) windows only.
        task_mask = window_task_mask(db, session_name, WINDOW_SIZE, BIN_WIDTH_SEC)
        if len(task_mask) != len(windows):
            # Would silently misalign the task mask with the windows.
            continue
        if uses_water(paradigm):
            n = len(windows) * WINDOW_SIZE
            # Linear maze: a window without a water position in every bin has no region.
            has_water = np.isfinite(water[:n]).reshape(len(windows), WINDOW_SIZE)
            task_mask = task_mask & has_water.all(axis=1)
        task_idx = np.flatnonzero(task_mask)
        if len(task_idx) < min_windows:
            continue
        if len(np.unique(regions[task_idx])) < min_classes:
            continue
        train_idx, val_idx, test_idx = chronological_task_split(
            task_mask,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

        session_data[session_name] = {
            "windows": windows,
            "regions": regions,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "x_positions": x_positions,
            "y_positions": y_positions,
            "water_positions": water,
        }
        valid_sessions.append(session_name)

    return session_data, valid_sessions, session_configs, shared_neuron_ids, n_shared


def normalize_session_windows(
    session_data: dict,
    valid_sessions: list[str],
) -> None:
    """Normalize windows per-unit using train-split statistics.

    Computes mean and std from training windows only, then z-scores all
    windows with those statistics, so no validation or test statistic ever
    reaches the encoder. The unnormalized counts are kept as ``windows_raw`` (the
    Poisson reconstruction target); an existing ``windows_raw`` is left as it is.

    Args:
        session_data:   assembled session dict from assemble_session_data; modified in
                        place.
        valid_sessions: sessions to normalize
    """
    for session_name in valid_sessions:
        d = session_data[session_name]
        if "windows_raw" not in d:
            d["windows_raw"] = d["windows"].copy()
        train_windows = d["windows"][d["train_idx"]]  # (n_train, T, n_neurons)
        mu = train_windows.mean(axis=(0, 1), keepdims=True)  # (1, 1, n_neurons)
        # The epsilon keeps a unit that is silent in training from dividing by zero.
        sd = train_windows.std(axis=(0, 1), keepdims=True) + 1e-3
        d["windows"] = ((d["windows"] - mu) / sd).astype("float32")


def force_session_specific(session_configs: dict) -> None:
    """Rewrite every session config to be 100% session-specific (n_shared = 0).

    Every neuron is then read through its session's own projection, and none through the
    shared one.

    Args:
        session_configs: ``build_hybrid_neuron_mapping`` configs; modified in place.
    """
    for cfg in session_configs.values():
        n = cfg["n_total"]
        cfg["specific_indices"] = list(range(n))
        cfg["specific_ids"] = list(cfg["all_ids"])
        cfg["shared_indices"] = []
        cfg["n_specific"] = n
        cfg["n_shared"] = 0


def apply_window_region_labels(
    session_data: dict,
    valid_sessions: list[str],
    paradigm: str,
    *,
    reduce: str,
    drop_straddle: bool,
    window_size: int = WINDOW_SIZE,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """Re-label each window by its modal region and optionally drop straddlers.

    Each session gets a ``straddle`` flag per window. Dropping removes straddling windows
    from the split indices only; the window arrays keep their length.

    Args:
        session_data:   assembled sessions; modified in place.
        valid_sessions: sessions to re-label.
        paradigm:       maze type string.
        reduce:         ``"mode"`` re-labels the windows; ``"last"`` keeps the labels
                        and only flags straddlers.
        drop_straddle:  remove straddling windows from the split indices.
        window_size:    bins per window.
        log_fn:         sink for per-session warnings (pass ``print`` to see them). A
                        session with fewer position bins than windows is left unchanged.

    Returns:
        ``(n_straddled, n_total)`` over the re-labeled sessions, for the caller to
        summarize.
    """
    straddled = total = 0
    for s in valid_sessions:
        d = session_data[s]
        new_reg, straddle = window_region_labels(
            d["x_positions"],
            d["y_positions"],
            window_size,
            paradigm,
            water=d.get("water_positions"),
            reduce=reduce,
        )
        n = len(d["regions"])
        if len(new_reg) < n:
            if log_fn is not None:
                log_fn(f"  warn {s}: position bins ({len(new_reg)}) < windows ({n}) ")
            continue
        new_reg, straddle = new_reg[:n], straddle[:n]
        straddled += int(straddle.sum())
        total += n
        if reduce != "last":
            d["regions"] = new_reg
        d["straddle"] = straddle
        if drop_straddle:
            keep = ~straddle
            for k in ("train_idx", "val_idx", "test_idx"):
                d[k] = d[k][keep[d[k]]]
    return straddled, total


def gate_sessions_conformal(
    session_data: dict,
    valid_sessions: list[str],
    min_val: int,
    min_test: int,
) -> list[tuple[str, int, int]]:
    """Drop sessions whose val/test block is too small to support a q_hat.

    ``val`` supplies the conformal quantile, so it needs enough windows for
    ``ceil((n+1)(1-alpha)) <= n``. ``test`` needs enough windows for coverage to
    mean something. Sessions failing either floor are removed from
    ``valid_sessions`` and popped from ``session_data``.

    Args:
        session_data:   assembled sessions; failing ones are popped.
        valid_sessions: session list; failing ones are removed in place.
        min_val:        minimum calibration (val) windows.
        min_test:       minimum test windows.

    Returns:
        The dropped ``(session, n_val, n_test)`` triples for the caller to report.
        Raising on an empty cohort is left to the caller.
    """
    dropped: list[tuple[str, int, int]] = []
    for s in list(valid_sessions):
        d = session_data[s]
        n_val, n_test = len(d["val_idx"]), len(d["test_idx"])
        if n_val < min_val or n_test < min_test:
            dropped.append((s, n_val, n_test))
            valid_sessions.remove(s)
            session_data.pop(s, None)
    return dropped


def assert_chronological_disjoint(
    session_data: dict, valid_sessions: list[str]
) -> None:
    """Assert train < val < test blocks are time-ordered and disjoint per session.

    Args:
        session_data:   assembled sessions.
        valid_sessions: sessions to check; each needs non-empty blocks.

    Raises:
        AssertionError: if a session's blocks overlap or are out of order.
    """
    for s in valid_sessions:
        d = session_data[s]
        tr, va, te = (np.sort(d[k]) for k in ("train_idx", "val_idx", "test_idx"))
        assert tr.max() < va.min() < va.max() < te.min(), (
            f"{s}: blocks are not chronological/disjoint: the split is broken"
        )


def compute_class_counts(
    session_data: dict,
    train_sessions: list[str],
) -> dict[int, int]:
    """Count training-set samples per region class across all training sessions.

    Used to build the class-balanced loss weights passed to train_step1 and
    train_step2.

    Args:
        session_data:    assembled session dict from assemble_session_data
        train_sessions:  session names whose train_idx rows are pooled

    Returns:
        {class_int: count_int} for the classes present; empty if no session has
        training windows.
    """
    parts = [
        session_data[s]["regions"][session_data[s]["train_idx"]]
        for s in train_sessions
        if s in session_data and len(session_data[s]["train_idx"]) > 0
    ]
    if not parts:
        return {}
    all_train_regions = np.concatenate(parts)
    unique_cls, cls_counts = np.unique(all_train_regions, return_counts=True)
    return {int(c): int(cnt) for c, cnt in zip(unique_cls, cls_counts, strict=False)}
