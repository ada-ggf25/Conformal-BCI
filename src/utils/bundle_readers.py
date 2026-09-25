"""Typed readers for the pooled per-maze ``metrics.json`` contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..conformal.trajectory import calibration_distance_trajectory, trajectory_summary
from ..data.config import BIN_WIDTH_SEC, WINDOW_SIZE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Name of the deployed method
_DEPLOYED_ARM = "learned-scale class-conditional"


def n_mice(bundle: dict) -> int:
    """Number of distinct mice in a pooled bundle, from its session names (M<id>_<date>)."""
    return len({s.split("_")[0] for s in bundle.get("sessions", [])})


def per_region_coverage_table(bundle: dict) -> dict[str, Any]:
    """Per-region test coverage and sharpness of the learned scale for a pooled bundle."""
    c = bundle.get("conformal") or {}
    prc = c.get("per_region_coverage") or {}
    lrn = prc.get("learned") or {}
    n_test = prc.get("n_test") or {}
    size = prc.get("size") or {}
    n_classes = bundle.get("n_classes")
    regions = sorted(lrn, key=lambda r: int(r))
    return {
        "regions": [int(r) for r in regions],
        "learned": [lrn.get(r) for r in regions],
        "n_test": [n_test.get(r) for r in regions],
        "size_over_k": [
            size[r] / n_classes if size.get(r) is not None and n_classes else None
            for r in regions
        ],
        "target": 1.0 - float(c.get("alpha", 0.10)),
    }


def region_distribution(bundle: dict) -> dict[str, Any]:
    """Per-region window counts across the entire dataset for a maze."""
    cc = bundle.get("class_counts") or {}
    regions = sorted(cc, key=lambda r: int(r))
    counts = [int(cc[r]) for r in regions]
    total = sum(counts)
    nan = float("nan")
    return {
        "regions": [int(r) for r in regions],
        "counts": counts,
        "fraction": [c / total if total else nan for c in counts],
    }


def operating_curves(bundle: dict) -> dict[str, Any]:
    """Coverage-sharpness frontier for a pooled bundle."""
    c = bundle.get("conformal") or {}
    k = int(bundle.get("n_classes") or 0)
    rows = (c.get("retrain_sweep") or {}).get("learned")
    rows = sorted(rows, key=lambda r: r["coverage"]) if rows else []
    return {
        "coverage": [float(r["coverage"]) for r in rows],
        "size_over_k": [float(r["size"]) / k if k else float("nan") for r in rows],
        "target": [float(r["target"]) for r in rows],
        "coverage_target": 1.0 - float(c.get("alpha", 0.10)),
        "n_classes": k,
    }


def whole_test_coverage(bundle: dict) -> dict[str, Any]:
    """Pooled whole-test coverage and its per-session spread for a pooled bundle."""
    c = bundle.get("conformal") or {}
    tc = c.get("test_coverage") or {}
    wt = tc.get("whole_test") or {}
    per = tc.get("per_session") or []
    return {
        "coverage": float(wt.get("coverage", float("nan"))),
        "n_windows": int(wt.get("n", 0)),
        "error": float(wt.get("error", float("nan"))),
        "meets_target": wt.get("meets_target"),
        "per_session_coverage": [float(r.get("coverage", float("nan"))) for r in per],
        "per_session_meets": [bool(r.get("meets_target", False)) for r in per],
        "n_sessions": len(per),
        "n_sessions_below": int(
            tc.get(
                "n_sessions_below_target",
                sum(not r.get("meets_target", False) for r in per),
            )
        ),
        "target": 1.0 - float(c.get("alpha", 0.10)),
    }


def class_conditional_thresholds(bundle: dict) -> dict[int, float]:
    """Per-region conformal thresholds (q_hat) for the deployed learned-scale."""
    arm = ((bundle.get("conformal") or {}).get("arms") or {}).get(_DEPLOYED_ARM) or {}
    raw = arm.get("q_hat_by_region") or {}
    return {int(c): (float("inf") if v is None else float(v)) for c, v in raw.items()}


def deployed_operating_point(bundle: dict) -> dict[str, float]:
    """Coverage and normalized set size of the deployed model."""
    nan = float("nan")
    n_classes = bundle.get("n_classes")
    arm = ((bundle.get("conformal") or {}).get("arms") or {}).get(_DEPLOYED_ARM) or {}
    set_size = arm.get("set_size")
    coverage = whole_test_coverage(bundle)["coverage"]
    return {
        "coverage": coverage,
        "size_over_k": (
            float(set_size) / n_classes if set_size is not None and n_classes else nan
        ),
    }


def maze_summary_table(mazes: dict, order: Sequence[str]) -> list[dict[str, Any]]:
    """One summary row per maze: the cross-maze comparison table.

    Pulls cohort size, decoder quality, whole-test coverage and the deployed operating
    point through the section readers above, so the table and the figures beside it can
    never disagree about a number.

    Args:
        mazes: paradigm -> bundle, as returned by ``load_maze_bundles``.
        order: paradigms to emit, in display order, unknown names are skipped.

    Returns:
        list of row dicts (``maze``, ``n_mice``, ``n_sessions``, ``n_classes``,
        ``test_windows``, ``raw_test_acc``, ``test_cov``, ``cov_meets_target``,
        ``n_sessions_below``, ``size_over_k``). Ready for ``pd.DataFrame``.
    """
    rows: list[dict[str, Any]] = []
    for p in order:
        m = mazes.get(p)
        if m is None:
            continue
        wt = whole_test_coverage(m)
        dep = deployed_operating_point(m)
        n_classes = m.get("n_classes")
        rows.append(
            {
                "maze": p,
                "n_mice": n_mice(m),
                "n_sessions": len(m.get("sessions", [])),
                "n_classes": n_classes if n_classes is not None else float("nan"),
                "test_windows": wt["n_windows"],
                "raw_test_acc": (m.get("decoder") or {}).get(
                    "raw_accuracy", float("nan")
                ),
                "test_cov": wt["coverage"],
                "cov_meets_target": wt["meets_target"],
                "n_sessions_below": wt["n_sessions_below"],
                "size_over_k": dep["size_over_k"],
            }
        )
    return rows


def trajectory_summary_for_bundle(
    bundle: dict,
    *,
    bin_width_sec: float = 20.0,
    min_sessions: int = 3,
) -> dict[str, float] | None:
    """``trajectory_summary`` computed directly from a pooled bundle's saved arrays.

    Reads ``_arrays_path``, which ``load_maze_bundles`` injects below.

    Returns:
        The ``trajectory_summary`` dict, or ``None`` if the bundle has no arrays, no
        finite per-region thresholds, or no recorded class count.
    """
    arr_path = bundle.get("_arrays_path")
    qhats = class_conditional_thresholds(bundle)
    n_classes = bundle.get("n_classes")
    if arr_path is None or not qhats or not n_classes:
        return None

    arrs = np.load(arr_path)
    test_mask = arrs["win_split"] == 2
    traj = calibration_distance_trajectory(
        arrs["win_probs"][test_mask],
        arrs["win_sigma"][test_mask],
        arrs["win_region"][test_mask],
        arrs["win_session_idx"][test_mask],
        qhats,
        bin_width_sec=bin_width_sec,
        seconds_per_window=WINDOW_SIZE * BIN_WIDTH_SEC,
        min_sessions=min_sessions,
    )
    return trajectory_summary(traj)


def load_maze_bundles(
    artifacts_dir: Path | str,
    maze_order: tuple[str, ...] = ("More_Water", "Y_Maze", "Nested_All"),
    *,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict, list[str]]:
    """Load every pooled per-maze bundle under ``<artifacts_dir>/<Paradigm>/ALL_MICE/metrics.json``.

    Args:
        artifacts_dir: the ``analyses/artifacts`` directory.
        maze_order:    preferred display order.
        log_fn:        optional sink for per-skip messages (pass ``print`` to see them).

    Returns:
        ``(mazes, order)``: ``mazes`` keyed by paradigm, ``order`` the display order.
    """
    artifacts_dir = Path(artifacts_dir)
    mazes: dict = {}
    for mf in sorted(artifacts_dir.glob("*/ALL_MICE/metrics.json")):
        m = json.loads(mf.read_text())
        c = m.get("conformal") or {}
        if "test_coverage" not in c:
            if log_fn is not None:
                log_fn(
                    f"  skip {mf.parent.parent.name}: pooled bundle missing conformal.test_coverage"
                )
            continue
        m["_arrays_path"] = (
            (mf.parent / "arrays.npz") if (mf.parent / "arrays.npz").exists() else None
        )
        m["_scale_net_path"] = (
            (mf.parent / "scale_net.pt")
            if (mf.parent / "scale_net.pt").exists()
            else None
        )
        mazes[m.get("paradigm", mf.parent.parent.name)] = m
    order = [p for p in maze_order if p in mazes] + [
        p for p in sorted(mazes) if p not in maze_order
    ]
    return mazes, order
