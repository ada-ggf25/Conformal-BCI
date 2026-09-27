#!/usr/bin/env python
"""Papermill runner for the per-maze pipeline, swept over seeds.

The ``per_maze.ipynb`` trains one model on every usable mouse in a maze.
The notebook runs once per (seed, maze), injecting ``TARGET_PARADIGM``, ``SEED``, that maze's
``EXCLUDE_SESSIONS`` and the output directory into its parameters cell.

Each run renders the executed notebook and writes the results bundle to
``analyses/artifacts/<Paradigm>/ALL_MICE/seed_<SEED>/``. Runs go seed by seed (every maze for
the first seed, then the next seed), so a partial sweep is already a complete set of mazes.
A (seed, maze) whose bundle already exists is skipped, and an interrupted (seed, maze) resumes
from the chain checkpoint the notebook writes after every session; ``--force`` reruns it from
scratch. ``--quick`` writes to ``analyses/artifacts/_quick/`` instead, so a smoke run never
stands in for a real bundle.

Examples:
    uv sync                                                          # installs papermill
    make maze                                                        # all 3 mazes, 5 seeds
    make maze PARADIGM=Y_Maze SEEDS="42 43"                          # one maze, two seeds
    uv run python analyses/run_mazes.py --paradigm More_Water Y_Maze # several mazes, in order
    uv run python analyses/run_mazes.py --quick                      # 2-epoch smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in _HERE.parents if (p / "pyproject.toml").exists()), _HERE.parent
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# deferred: needs REPO_ROOT on sys.path first
from src.utils.artifacts import paradigm_slug  # noqa: E402
from src.utils.cohort import MERGED  # noqa: E402

TEMPLATE = REPO_ROOT / "analyses" / "per_maze.ipynb"
ARTIFACTS = REPO_ROOT / "analyses" / "artifacts"
QUICK_ARTIFACTS = ARTIFACTS / "_quick"

# the seed sweep: every seed runs the whole pipeline
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# the three analyzed mazes. A run `make maze` trains all three in order
DEFAULT_PARADIGMS = ["More_Water", "Y_Maze", "Nested_All"]

# Manual per-maze session exclusions (data curation). Session ids listed here are discarded
# before the pipeline loads them, via the notebook's EXCLUDE_SESSIONS papermill parameter.
# Criterion is recording quality.
EXCLUDE_SESSIONS: dict[str, list[str]] = {
    "More_Water": [
        "M46_250512",
        "M46_250424",
        "M25_241004",
        "M27_240920",
        "M19_241023",
        "M19_240922",
        "M21_241018",
        "M19_241018",
        "M21_241005",
    ],
    "Y_Maze": ["M47_250410", "M27_250202"],
    "Nested_All": ["M27_250105", "M19_241125"],
}


def bundle_dir(root: Path, paradigm: str, seed: int) -> Path:
    """Where one (maze, seed) run writes its bundle and executed notebook."""
    return root / paradigm_slug(paradigm) / "ALL_MICE" / f"seed_{seed}"


def is_complete(out_dir: Path) -> bool:
    """A bundle is complete once its last-written file, ``arrays.npz``, exists."""
    return (out_dir / "metrics.json").exists() and (out_dir / "arrays.npz").exists()


def run_one(paradigm: str, out_dir: Path, params: dict) -> None:
    """Execute the template for one (maze, seed) into ``out_dir``.

    ``paradigm`` is a concrete paradigm string ("More_Water"): it is passed straight through
    as ``TARGET_PARADIGM``.
    """
    import papermill as pm

    out_dir.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        str(TEMPLATE),
        str(out_dir / "per_maze.ipynb"),
        parameters={
            "TARGET_PARADIGM": paradigm,
            "ARTIFACT_DIR": str(out_dir),
            **params,
        },
        cwd=str(REPO_ROOT),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paradigm",
        nargs="*",
        default=DEFAULT_PARADIGMS,
        help=f"maze(s) to pool + train; default {DEFAULT_PARADIGMS}. "
        f"Merged presets: {sorted(MERGED)}",
    )
    ap.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=f"seeds to sweep; default {DEFAULT_SEEDS} ({DEFAULT_SEEDS[:1]} with --quick)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: 2-epoch models, a 5-epoch sigma and a 2-point coverage sweep, "
        "written under analyses/artifacts/_quick/",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rerun (seed, maze) pairs whose bundle already exists",
    )
    args = ap.parse_args()

    seeds = args.seeds or (DEFAULT_SEEDS[:1] if args.quick else DEFAULT_SEEDS)
    root = QUICK_ARTIFACTS if args.quick else ARTIFACTS
    quick: dict = (
        dict(
            STEP_EPOCHS=2,
            EARLY_STOP_PATIENCE=2,
            MLP_EPOCHS=5,
            MLP_EARLY_STOP_PATIENCE=3,
            SWEEP_TARGETS=[0.5, 0.9],
        )
        if args.quick
        else {}
    )

    ok, skipped, failed = [], [], []
    for seed in seeds:
        for pg in args.paradigm:
            out_dir = bundle_dir(root, pg, seed)
            tag = f"{pg} seed {seed}"
            if is_complete(out_dir) and not args.force:
                print(f"\n=== {tag}: bundle exists, skipped ({out_dir})")
                skipped.append(tag)
                continue
            print(f"\n=== {tag} ===")
            if args.force:  # start over, not from an interrupted run's checkpoint
                (out_dir / "checkpoint.pt").unlink(missing_ok=True)
            try:
                run_one(
                    pg,
                    out_dir,
                    {
                        **quick,
                        "SEED": seed,
                        "EXCLUDE_SESSIONS": EXCLUDE_SESSIONS.get(pg, []),
                    },
                )
                print(f"  OK -> {out_dir}")
                ok.append(tag)
            except Exception as exc:  # a broken run must not abort the others
                print(f"  FAIL: {type(exc).__name__}: {exc}")
                failed.append(tag)

    print(
        f"\n{len(ok)} OK | {len(skipped)} skipped (already done) | {len(failed)} failed"
        + (f": {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
