"""Filesystem layout and write side of the per-maze analysis bundles (one per seed).

A bundle is ``metrics.json`` (summary numbers), ``arrays.npz`` (one row per tested window)
and ``sigma_nets.pt`` (each tested session's learned scale). ``bundle_readers`` is the read
side.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..conformal.signals import SIGNALS


def paradigm_slug(paradigm: str) -> str:
    """Filesystem-safe directory name for a paradigm string.

    Runs of non-alphanumeric characters collapse to a single underscore, so
    ``'Nested_Maze+2'`` becomes ``'Nested_Maze_2'``.

    Args:
        paradigm: paradigm string or cohort label.

    Returns:
        The slug, with leading and trailing underscores stripped.

    Raises:
        ValueError: if paradigm is empty or has no alphanumeric characters.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", paradigm).strip("_")
    if not slug:
        raise ValueError(
            f"paradigm {paradigm!r} yields an empty directory name; a concrete "
            "paradigm must be resolved before an artifact path is built"
        )
    return slug


def jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types.

    Numpy floats/ints/bools become Python scalars, arrays become nested lists, and
    non-finite floats (``inf`` / ``nan``) become ``None`` so ``json.dump`` succeeds.
    Dict keys become strings and tuples become lists.

    Args:
        obj: any nesting of dicts, lists, tuples, numpy values and Python scalars.

    Returns:
        The same structure built from JSON-serializable types only.
    """
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


DEPLOYED_ARM = "learned-scale class-conditional"
"""Name of the deployed method in ``metrics.json['conformal']['arms']``."""

LEVEL_CODES = {"own": 0, "borrowed": 1, "global": 2}
"""Integer codes for the threshold level stored per test window in ``arrays.npz``."""

GROUPS = ("all", "returning", "first")
"""Session groups summarized in ``metrics.json``: every tested session, returning mice, and the
first session of each mouse."""

MIN_REGION_WINDOWS = 5
"""Per-region coverage and size are reported only for regions with this many test windows."""


def _concat(rows: list[dict], method: str, key: str) -> np.ndarray:
    """One method's per-window ``key`` array, concatenated over rows in order."""
    return np.concatenate([r["methods"][method][key] for r in rows])


def method_summary(rows: list[dict], method: str, target: float) -> dict[str, Any]:
    """One method's results over a group of tested sessions, pooled over test windows.

    ``mean_abs_gap`` and ``sessions_below`` are per-session diagnostics against ``target``.

    Args:
        rows:   ``run_chain`` rows of the group's sessions.
        method: a key of each row's ``methods``.
        target: the coverage target ``1 - alpha``.

    Returns:
        ``n_sessions``, ``n_windows``, pooled ``coverage``, ``size_over_k`` and
        ``accuracy``, the per-session ``mean_abs_gap`` and ``sessions_below``, and the
        share of windows whose threshold was ``own``, ``borrowed`` or ``global``.
    """
    covered = _concat(rows, method, "covered")
    per_session = np.array([r["methods"][method]["covered"].mean() for r in rows])
    level = _concat(rows, method, "level")
    return {
        "n_sessions": len(rows),
        "n_windows": len(covered),
        "coverage": float(covered.mean()),
        "size_over_k": float(_concat(rows, method, "size_over_k").mean()),
        "accuracy": float(1.0 - _concat(rows, method, "error").mean()),
        "mean_abs_gap": float(np.mean(np.abs(per_session - target))),
        "sessions_below": int((per_session < target).sum()),
        "share_own": float((level == "own").mean()),
        "share_borrowed": float((level == "borrowed").mean()),
        "share_global": float((level == "global").mean()),
    }


def build_chain_bundle(
    rows: list[dict],
    *,
    chain: list[str],
    n_warmup: int,
    session_data: dict,
    n_classes: int,
    paradigm: str,
    seed: int,
    alpha: float,
    rule: str,
    config: dict,
    training: list[dict] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """``metrics.json`` and ``arrays.npz`` contents for one chain run.

    ``metrics.json`` keeps the keys the cross-maze readers read (``decoder``, and
    ``conformal`` ``arms`` / ``test_coverage`` / ``per_region_coverage`` / ``retrain_sweep``),
    all for the deployed method, pooled over the test windows of every tested session;
    ``conformal.methods`` adds every method per session group; ``training`` holds every session's
    training curves. ``arrays.npz`` has one row per tested window, in chain order, and, when the
    rows carry them, the test windows' probabilities and sigma plus a calibration table
    (``cal_*``, one row per calibration window of every tested session, own block and the
    re-scored past blocks), enough to recompute any calibration rule offline.

    Args:
        rows:         ``run_chain`` output, one row per tested session.
        chain:        every session of the chain, in order (warm-up first).
        n_warmup:     how many leading sessions were warm-up.
        session_data: the prepared sessions, for the region distribution.
        n_classes:    number of regions K.
        paradigm:     the maze's paradigm label.
        seed:         the run's seed.
        alpha:        miscoverage level; the coverage target is ``1 - alpha``.
        rule:         the deployed pool rule (``"B3"``); ``chain_<rule>`` is the headline method.
        config:       the run's configuration, recorded as given.
        training:     ``run_chain``'s training log, recorded as given.

    Returns:
        ``(metrics, arrays)``: the ``metrics.json`` dict (not yet passed through
        ``jsonify``) and the ``arrays.npz`` arrays. Per-window arrays are named
        ``win_<field>`` or ``win_<field>_<method>``; the calibration table's arrays are
        ``cal_session_idx`` (the tested session, as ``win_session_idx``), ``cal_offset``
        (0 = its own block, k = k sessions back), ``cal_region``, ``cal_score`` and
        ``cal_sigma``. Also ``win_window_idx`` (the window's index within its session),
        ``sweep_targets`` with ``win_covered_sweep`` / ``win_size_over_k_sweep`` (one column
        per target), and, from ``training``, the windows every chain session's sigma was fit on
        (``fit_chain_idx`` into ``metrics['sessions']``, ``fit_block`` 0 = train and
        1 = calibration, ``fit_region``, ``fit_probs``, ``fit_signals`` with columns in
        ``SIGNALS`` order, raw).
    """
    target = 1.0 - alpha
    main = f"chain_{rule}"
    deployed = [r["methods"][main] for r in rows]
    labels = np.concatenate([r["labels"] for r in rows])
    covered = _concat(rows, main, "covered")
    size = _concat(rows, main, "size_over_k") * n_classes
    error = _concat(rows, main, "error")

    per_session = []
    for r, ev in zip(rows, deployed, strict=True):
        cov = float(ev["covered"].mean())
        per_session.append(
            {
                "session": r["session"],
                "mouse": r["mouse"],
                "first_of_mouse": bool(r["first_of_mouse"]),
                "n_neurons": r["n_neurons"],
                "n": len(ev["covered"]),
                "coverage": cov,
                "error": float(ev["error"].mean()),
                "size_over_k": float(ev["size_over_k"].mean()),
                "meets_target": bool(cov >= target),
                "share_global": float((ev["level"] == "global").mean()),
            }
        )

    region_n = np.bincount(labels, minlength=n_classes)
    # Regions with too few test windows are reported as None rather than a noisy mean.
    per_region: dict[str, dict[int, Any]] = {"learned": {}, "n_test": {}, "size": {}}
    for c in range(n_classes):
        m = labels == c
        enough = region_n[c] >= MIN_REGION_WINDOWS
        per_region["learned"][c] = float(covered[m].mean()) if enough else None
        per_region["size"][c] = float(size[m].mean()) if enough else None
        per_region["n_test"][c] = int(region_n[c])

    sweep = []
    for t in rows[0]["sweep"]:
        cov_t = np.concatenate([r["sweep"][t]["covered"] for r in rows])
        size_t = np.concatenate([r["sweep"][t]["size_over_k"] for r in rows])
        sweep.append(
            {
                "target": float(t),
                "coverage": float(cov_t.mean()),
                "size": float(size_t.mean() * n_classes),
                "deployed": bool(np.isclose(t, target)),
            }
        )

    groups = {
        "all": rows,
        "returning": [r for r in rows if not r["first_of_mouse"]],
        "first": [r for r in rows if r["first_of_mouse"]],
    }
    methods = {
        g: {m: method_summary(grp, m, target) for m in rows[0]["methods"]}
        for g, grp in groups.items()
        if grp
    }

    # Region distribution over every task window of every chain session, warm-up included.
    counts = np.zeros(n_classes, dtype=int)
    for s in chain:
        d = session_data[s]
        task = np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]])
        counts += np.bincount(d["regions"][task], minlength=n_classes)

    metrics = {
        "paradigm": paradigm,
        "seed": seed,
        "sessions": list(chain),
        "tested_sessions": [r["session"] for r in rows],
        "n_classes": int(n_classes),
        "class_counts": {c: int(v) for c, v in enumerate(counts)},
        "chain": {"warmup": list(chain[:n_warmup]), "config": config},
        "decoder": {
            "raw_accuracy": float(1.0 - error.mean()),
            "chance": 1.0 / n_classes,
            "majority_baseline": float(region_n.max() / region_n.sum()),
        },
        "conformal": {
            "alpha": alpha,
            "rule": rule,
            "arms": {DEPLOYED_ARM: {"set_size": float(size.mean())}},
            "test_coverage": {
                "whole_test": {
                    "coverage": float(covered.mean()),
                    "n": len(covered),
                    "error": float(error.mean()),
                    "meets_target": bool(covered.mean() >= target),
                },
                "per_session": per_session,
                "n_sessions_below_target": sum(
                    not r["meets_target"] for r in per_session
                ),
            },
            "per_region_coverage": per_region,
            "retrain_sweep": {"learned": sweep},
            "methods": methods,
        },
    }

    arrays: dict[str, np.ndarray] = {
        "win_session_idx": np.concatenate(
            [np.full(len(r["labels"]), i) for i, r in enumerate(rows)]
        ),
        "win_region": labels.astype(int),
        "win_sigma_rel": np.concatenate([r["sigma_rel"] for r in rows]),
        "win_level": np.array([LEVEL_CODES[v] for v in _concat(rows, main, "level")]),
    }
    for m in rows[0]["methods"]:
        for key in ("covered", "size_over_k", "error"):
            arrays[f"win_{key}_{m}"] = _concat(rows, m, key)
    u = np.concatenate([r["signals"] for r in rows])
    for j, name in enumerate(SIGNALS):
        arrays[f"win_{name}"] = u[:, j]
    if all("probs" in r for r in rows):
        arrays["win_probs"] = np.concatenate([r["probs"] for r in rows])
        arrays["win_sigma"] = np.concatenate([r["sigma"] for r in rows])
    if all("calibration" in r for r in rows):
        blocks = [(i, c) for i, r in enumerate(rows) for c in r["calibration"]]
        arrays["cal_session_idx"] = np.concatenate(
            [np.full(len(c["labels"]), i) for i, c in blocks]
        )
        arrays["cal_offset"] = np.concatenate(
            [np.full(len(c["labels"]), c["offset"]) for _, c in blocks]
        )
        arrays["cal_region"] = np.concatenate([c["labels"] for _, c in blocks]).astype(
            int
        )
        arrays["cal_score"] = np.concatenate([c["scores"] for _, c in blocks])
        arrays["cal_sigma"] = np.concatenate([c["sigma"] for _, c in blocks])
    if all("test_idx" in r for r in rows):
        arrays["win_window_idx"] = np.concatenate([r["test_idx"] for r in rows])
    targets = list(rows[0]["sweep"])
    arrays["sweep_targets"] = np.array(targets, dtype=float)
    for key in ("covered", "size_over_k"):
        arrays[f"win_{key}_sweep"] = np.stack(
            [np.concatenate([r["sweep"][t][key] for r in rows]) for t in targets],
            axis=1,
        )
    if training is not None:
        fits = [
            (i, e["fit_inputs"]) for i, e in enumerate(training) if "fit_inputs" in e
        ]
        if fits:
            fit_blocks = [
                (i, b, f[k]) for i, f in fits for b, k in enumerate(("tr", "va"))
            ]
            arrays["fit_chain_idx"] = np.concatenate(
                [np.full(len(x["labels"]), i) for i, _, x in fit_blocks]
            )
            arrays["fit_block"] = np.concatenate(
                [np.full(len(x["labels"]), b) for _, b, x in fit_blocks]
            )
            arrays["fit_region"] = np.concatenate(
                [x["labels"] for *_, x in fit_blocks]
            ).astype(int)
            arrays["fit_probs"] = np.concatenate([x["probs"] for *_, x in fit_blocks])
            arrays["fit_signals"] = np.concatenate([x["u"] for *_, x in fit_blocks])
        metrics["training"] = [
            {k: v for k, v in e.items() if k != "fit_inputs"} for e in training
        ]
    return metrics, arrays


def sigma_nets_payload(rows: list[dict], hidden: int, depth: int) -> dict[str, Any]:
    """The ``sigma_nets.pt`` contents: each tested session's sigma and its signal standardization.

    Session ``i`` here is row ``i`` of ``run_chain`` (``win_session_idx == i`` in ``arrays.npz``).

    Args:
        rows:   ``run_chain`` rows.
        hidden: the scale net's hidden width, needed to rebuild it.
        depth:  the scale net's depth, needed to rebuild it.

    Returns:
        ``{"signals", "hidden", "depth", "sessions"}``, one ``sessions`` entry per row with
        its name, state dict, and signal mean and std.
    """
    return {
        "signals": list(SIGNALS),
        "hidden": hidden,
        "depth": depth,
        "sessions": [
            {
                "session": r["session"],
                "state_dict": r["sigma_state"],
                "signal_mean": list(r["signal_stats"][0]),
                "signal_std": list(r["signal_stats"][1]),
            }
            for r in rows
        ],
    }
