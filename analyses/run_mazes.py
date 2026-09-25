#!/usr/bin/env python
"""Papermill runner for the per-maze pipeline.

The ``per_maze.ipynb`` trains one model on every usable mouse in a maze.
The notebook runs once per maze, injecting ``TARGET_PARADIGM`` and that maze's
``EXCLUDE_SESSIONS`` into its parameters cell.

Each run renders to ``analyses/artifacts/<Paradigm>/per_maze.ipynb`` and the notebook's save
cell writes the results bundle to ``analyses/artifacts/<Paradigm>/ALL_MICE/``.

Examples:
    uv sync                                                          # installs papermill
    make maze                                                        # pooled model, all 3 mazes
    make maze PARADIGM=Y_Maze                                        # pooled model, a single maze
    uv run python analyses/run_mazes.py --paradigm More_Water Y_Maze # several mazes, in order
    uv run python analyses/run_mazes.py --quick                      # 2/2-epoch smoke test
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


def run_one(paradigm: str, params: dict) -> None:
    """Execute the template for one maze -> artifacts/<paradigm_slug>/per_maze.ipynb.

    ``paradigm`` is a concrete paradigm string ("More_Water"): it is passed straight through
    as ``TARGET_PARADIGM``.

    The output dir is keyed by ``paradigm_slug(paradigm)``. The same name derieved for
    the notebook's save cell path.
    """
    import papermill as pm

    out_dir = ARTIFACTS / paradigm_slug(paradigm)
    out_dir.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        str(TEMPLATE),
        str(out_dir / "per_maze.ipynb"),
        parameters={"TARGET_PARADIGM": paradigm, **params},
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
        "--quick",
        action="store_true",
        help="smoke run: 2/2 model epochs and a 5-epoch conformal scale",
    )
    args = ap.parse_args()

    quick: dict = (
        dict(STEP1_EPOCHS=2, STEP2_EPOCHS=2, MLP_EPOCHS=5, MLP_EARLY_STOP_PATIENCE=3)
        if args.quick
        else {}
    )

    failed = []
    for pg in args.paradigm:
        print(f"\n=== Maze {pg} ===")
        try:
            run_one(pg, {**quick, "EXCLUDE_SESSIONS": EXCLUDE_SESSIONS.get(pg, [])})
            print(f"  OK -> {ARTIFACTS / paradigm_slug(pg) / 'ALL_MICE'}")
        except Exception as exc:  # a broken maze must not abort the others
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            failed.append(pg)

    n_ok = len(args.paradigm) - len(failed)
    print(
        f"\n{n_ok} OK | {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
