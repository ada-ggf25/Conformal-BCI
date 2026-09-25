"""Typed readers for the pooled per-maze ``metrics.json`` contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..conformal.error_correlation import error_correlations

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


def seed_values(runs: dict[int, dict], fn: Callable[[dict], Any]) -> list[Any]:
    """``fn`` applied to each seed's bundle, in seed order, dropping ``None`` results.

    Args:
        runs: seed -> bundle, one maze's entry of ``load_maze_bundles``.
        fn:   a per-bundle reader.
    """
    out = [fn(runs[s]) for s in sorted(runs)]
    return [v for v in out if v is not None]


def mean_sd(values: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise mean and sample sd (ddof=1) across seeds, ignoring None and NaN.

    ``values`` is one scalar or one equal-length sequence per seed. The sd is NaN where
    fewer than two seeds have a value, the mean where none has.
    """
    rows = [
        np.array([np.nan if v is None else v for v in np.atleast_1d(x)], dtype=float)
        for x in values
    ]
    if not rows:
        return np.array([np.nan]), np.array([np.nan])
    a = np.vstack(rows)
    n = np.sum(np.isfinite(a), axis=0)
    total = np.nansum(a, axis=0)
    mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    dev = np.where(np.isfinite(a), (a - mean) ** 2, 0.0).sum(axis=0)
    sd = np.where(n > 1, np.sqrt(dev / np.maximum(n - 1, 1)), np.nan)
    return mean, sd


def maze_summary_table(mazes: dict, order: Sequence[str]) -> list[dict[str, Any]]:
    """One summary row per maze: the cross-maze comparison table, aggregated over seeds.

    Pulls cohort size, decoder quality, whole-test coverage and the deployed operating
    point through the section readers above, so the table and the figures beside it can
    never disagree about a number. The cohort columns are identical across seeds (the data
    and the split do not depend on the seed) and come from the first seed.

    Args:
        mazes: paradigm -> {seed: bundle}, as returned by ``load_maze_bundles``.
        order: paradigms to emit, in display order, unknown names are skipped.

    Returns:
        list of row dicts (``maze``, ``n_seeds``, ``n_mice``, ``n_sessions``, ``n_classes``,
        ``test_windows``, and the mean and ``*_sd`` of ``raw_test_acc``, ``test_cov`` and
        ``size_over_k``). Ready for ``pd.DataFrame``.
    """
    rows: list[dict[str, Any]] = []
    for p in order:
        runs = mazes.get(p)
        if not runs:
            continue
        first = runs[min(runs)]
        n_classes = first.get("n_classes")
        row: dict[str, Any] = {
            "maze": p,
            "n_seeds": len(runs),
            "n_mice": n_mice(first),
            "n_sessions": len(first.get("sessions", [])),
            "n_classes": n_classes if n_classes is not None else float("nan"),
            "test_windows": whole_test_coverage(first)["n_windows"],
        }
        per_seed = {
            "raw_test_acc": seed_values(
                runs,
                lambda m: (m.get("decoder") or {}).get("raw_accuracy", float("nan")),
            ),
            "test_cov": seed_values(runs, lambda m: whole_test_coverage(m)["coverage"]),
            "size_over_k": seed_values(
                runs, lambda m: deployed_operating_point(m)["size_over_k"]
            ),
        }
        for k, v in per_seed.items():
            mean, sd = mean_sd(v)
            row[k], row[f"{k}_sd"] = float(mean[0]), float(sd[0])
        rows.append(row)
    return rows


def seed_session_coverage(runs: dict[int, dict]) -> dict[str, Any]:
    """Per-session test coverage averaged over seeds, matched by session name.

    Returns:
        ``{"sessions", "coverage", "meets", "target"}``: one entry per session, ``meets``
        being the seed-mean coverage at or above the target.
    """
    by_session: dict[str, list[float]] = {}
    for s in sorted(runs):
        tc = (runs[s].get("conformal") or {}).get("test_coverage") or {}
        for r in tc.get("per_session") or []:
            by_session.setdefault(r["session"], []).append(float(r["coverage"]))
    target = whole_test_coverage(runs[min(runs)])["target"]
    names = list(by_session)
    cov = [float(np.mean(by_session[n])) for n in names]
    return {
        "sessions": names,
        "coverage": cov,
        "meets": [c >= target for c in cov],
        "target": target,
    }


def seed_operating_curves(runs: dict[int, dict]) -> dict[str, Any]:
    """The retrained coverage-sharpness sweep, mean and sd over seeds, per target.

    Seeds are matched by coverage target (every seed sweeps the same grid). A target
    missing from a seed counts as missing, not as zero.

    Returns:
        ``{"target", "coverage", "coverage_sd", "size_over_k", "size_over_k_sd",
        "coverage_target"}``, sorted by target.
    """
    per_seed = [operating_curves(runs[s]) for s in sorted(runs)]
    targets = sorted({t for oc in per_seed for t in oc["target"]})

    def _aligned(key: str) -> list[list[float]]:
        out = []
        for oc in per_seed:
            by_t = dict(zip(oc["target"], oc[key], strict=True))
            out.append([by_t.get(t, float("nan")) for t in targets])
        return out

    out: dict[str, Any] = {
        "target": targets,
        "coverage_target": per_seed[0]["coverage_target"],
    }
    for key in ("coverage", "size_over_k"):
        mean, sd = mean_sd(_aligned(key)) if targets else (np.array([]), np.array([]))
        out[key], out[f"{key}_sd"] = mean.tolist(), sd.tolist()
    return out


def error_correlations_for_bundle(bundle: dict) -> dict | None:
    """``error_correlations`` over a pooled bundle's saved test windows.

    Reads ``_arrays_path``, which ``load_maze_bundles`` injects below.

    Returns:
        The ``error_correlations`` dict, or ``None`` if the bundle has no arrays, no
        finite per-region thresholds, or no recorded class count.
    """
    arr_path = bundle.get("_arrays_path")
    qhats = class_conditional_thresholds(bundle)
    if arr_path is None or not qhats or not bundle.get("n_classes"):
        return None

    arrs = np.load(arr_path)
    test_mask = arrs["win_split"] == 2
    return error_correlations(
        arrs["win_probs"][test_mask],
        arrs["win_sigma"][test_mask],
        arrs["win_region"][test_mask],
        qhats,
    )


def _load_bundle(metrics_path: Path) -> dict:
    """One bundle's ``metrics.json`` plus the paths of its saved arrays and scale net."""
    m = json.loads(metrics_path.read_text())
    d = metrics_path.parent
    m["_arrays_path"] = (d / "arrays.npz") if (d / "arrays.npz").exists() else None
    m["_scale_net_path"] = (
        (d / "scale_net.pt") if (d / "scale_net.pt").exists() else None
    )
    return m


def load_maze_bundles(
    artifacts_dir: Path | str,
    maze_order: tuple[str, ...] = ("More_Water", "Y_Maze", "Nested_All"),
    *,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[dict, list[str]]:
    """Load every per-seed bundle under ``<artifacts_dir>/<Paradigm>/ALL_MICE/seed_<k>/``.

    A maze with no ``seed_<k>`` directories falls back to a single bundle directly under
    ``ALL_MICE/`` (the layout before seed sweeps), keyed by its recorded seed or, for bundles
    that predate the ``seed`` field, by 42, the notebook's default.

    Args:
        artifacts_dir: the ``analyses/artifacts`` directory.
        maze_order:    preferred display order.
        log_fn:        optional sink for per-skip messages (pass ``print`` to see them).

    Returns:
        ``(mazes, order)``: ``mazes`` maps paradigm -> {seed: bundle}, ``order`` is the
        display order.
    """
    artifacts_dir = Path(artifacts_dir)
    mazes: dict = {}
    for cohort_dir in sorted(artifacts_dir.glob("*/ALL_MICE")):
        paths = sorted(cohort_dir.glob("seed_*/metrics.json"))
        if not paths and (cohort_dir / "metrics.json").exists():
            paths = [cohort_dir / "metrics.json"]
        for mf in paths:
            m = _load_bundle(mf)
            if "test_coverage" not in (m.get("conformal") or {}):
                if log_fn is not None:
                    log_fn(
                        f"  skip {mf.parent}: bundle missing conformal.test_coverage"
                    )
                continue
            seed = int(m.get("seed", 42))
            mazes.setdefault(m.get("paradigm", cohort_dir.parent.name), {})[seed] = m
    order = [p for p in maze_order if p in mazes] + [
        p for p in sorted(mazes) if p not in maze_order
    ]
    return mazes, order
