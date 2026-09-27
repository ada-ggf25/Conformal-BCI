"""Tests for analyses/run_mazes.py, the resumable seed x maze sweep.

``run_one`` (which executes the notebook) is replaced with a recorder, so these tests check
the sweep's bookkeeping: where each bundle goes, which runs are skipped as done, what a
forced rerun deletes, that one failure does not stop the rest, and what each run is given.

Accepted gaps: ``run_one`` itself (a thin papermill call; running the notebook needs the live
database).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "analyses" / "run_mazes.py"
_spec = importlib.util.spec_from_file_location("run_mazes", _PATH)
assert _spec is not None and _spec.loader is not None
run_mazes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_mazes)


def _complete(d: Path) -> None:
    """Write the files a finished run leaves behind."""
    d.mkdir(parents=True, exist_ok=True)
    for name in ("sigma_nets.pt", "metrics.json", "arrays.npz"):
        (d / name).write_bytes(b"")


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """Run main() with the given CLI args against tmp artifact roots; record every run."""
    calls: list[tuple[str, Path, dict]] = []
    fail: set[tuple[str, int]] = set()

    def fake_run_one(paradigm, out_dir, params):
        """Record the run, or fail it if asked to."""
        calls.append((paradigm, out_dir, params))
        if (paradigm, params["SEED"]) in fail:
            raise RuntimeError("boom")

    monkeypatch.setattr(run_mazes, "run_one", fake_run_one)
    monkeypatch.setattr(run_mazes, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(run_mazes, "QUICK_ARTIFACTS", tmp_path / "artifacts" / "_quick")

    def run(*args: str) -> int:
        """Invoke main() with these command-line arguments."""
        monkeypatch.setattr(sys, "argv", ["run_mazes.py", *args])
        return run_mazes.main()

    run.calls, run.fail, run.root = calls, fail, tmp_path / "artifacts"  # type: ignore[attr-defined]
    return run


class TestBundleDir:
    """Where a run's bundle goes: the cross-maze loader finds it only there."""

    def test_layout_is_paradigm_slug_then_seed(self, tmp_path):
        """A bundle lives at <root>/<paradigm slug>/ALL_MICE/seed_<k>, where the cross-maze
        loader looks for it."""
        assert run_mazes.bundle_dir(tmp_path, "Y_Maze", 42) == (
            tmp_path / "Y_Maze" / "ALL_MICE" / "seed_42"
        )

    def test_paradigm_is_made_filesystem_safe(self, tmp_path):
        """A paradigm with a "+" gets the same safe name the notebook writes to."""
        out = run_mazes.bundle_dir(tmp_path, "Nested_Maze+2", 7)
        assert out.relative_to(tmp_path).parts == (
            "Nested_Maze_2",
            "ALL_MICE",
            "seed_7",
        )


class TestIsComplete:
    """A bundle counts as done only once the notebook has written it in full."""

    def test_a_fully_written_bundle_is_complete(self, tmp_path):
        """metrics.json plus arrays.npz (written last by the notebook) marks it done."""
        _complete(tmp_path)
        assert run_mazes.is_complete(tmp_path)

    @pytest.mark.parametrize(
        "present",
        [(), ("checkpoint.pt",), ("sigma_nets.pt",), ("sigma_nets.pt", "metrics.json"),
         ("arrays.npz",)],
    )  # fmt: skip
    def test_a_partial_bundle_is_not_complete(self, tmp_path, present):
        """A run interrupted before or during saving must be rerun, not skipped."""
        for name in present:
            (tmp_path / name).write_bytes(b"")
        assert not run_mazes.is_complete(tmp_path)


class TestMain:
    """The sweep's order, skipping, resuming, and failure handling."""

    def test_runs_go_seed_by_seed_through_every_maze(self, sweep):
        """Every maze runs for the first seed before the next seed starts, so a partial sweep
        already holds complete sets of mazes."""
        assert sweep("--paradigm", "Y_Maze", "More_Water", "--seeds", "42", "43") == 0
        order = [(p, params["SEED"]) for p, _, params in sweep.calls]
        assert order == [
            ("Y_Maze", 42),
            ("More_Water", 42),
            ("Y_Maze", 43),
            ("More_Water", 43),
        ]

    def test_defaults_sweep_every_maze_and_seed(self, sweep):
        """With no arguments the full sweep runs: 3 mazes x 5 seeds, into the real artifact root."""
        sweep()
        assert len(sweep.calls) == len(run_mazes.DEFAULT_PARADIGMS) * len(
            run_mazes.DEFAULT_SEEDS
        )
        assert all(out.is_relative_to(sweep.root) for _, out, _ in sweep.calls)
        assert not any("_quick" in out.parts for _, out, _ in sweep.calls)

    def test_finished_bundles_are_skipped(self, sweep):
        """A (maze, seed) with a complete bundle is not rerun; an unfinished one is."""
        _complete(run_mazes.bundle_dir(sweep.root, "Y_Maze", 42))
        sweep("--paradigm", "Y_Maze", "--seeds", "42", "43")
        assert [params["SEED"] for _, _, params in sweep.calls] == [43]

    def test_an_interrupted_run_keeps_its_checkpoint(self, sweep):
        """Without --force an unfinished run resumes: its chain checkpoint is left in place."""
        d = run_mazes.bundle_dir(sweep.root, "Y_Maze", 42)
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_bytes(b"ckpt")
        sweep("--paradigm", "Y_Maze", "--seeds", "42")
        assert (d / "checkpoint.pt").exists() and len(sweep.calls) == 1

    def test_force_reruns_from_scratch(self, sweep):
        """--force reruns a finished bundle and deletes any checkpoint, so nothing resumes."""
        d = run_mazes.bundle_dir(sweep.root, "Y_Maze", 42)
        _complete(d)
        (d / "checkpoint.pt").write_bytes(b"ckpt")
        sweep("--paradigm", "Y_Maze", "--seeds", "42", "--force")
        assert len(sweep.calls) == 1 and not (d / "checkpoint.pt").exists()

    def test_a_failed_run_does_not_stop_the_rest(self, sweep, capsys):
        """One broken (maze, seed) is reported and the sweep carries on; the exit code is 1."""
        sweep.fail.add(("Y_Maze", 42))
        code = sweep("--paradigm", "Y_Maze", "More_Water", "--seeds", "42", "43")
        assert code == 1 and len(sweep.calls) == 4
        assert "1 failed: Y_Maze seed 42" in capsys.readouterr().out

    def test_each_run_gets_its_seed_exclusions_and_output_dir(self, sweep):
        """A run is told its seed, that maze's curated exclusions, and its own bundle dir."""
        sweep("--paradigm", "More_Water", "Nested_All", "--seeds", "44")
        for paradigm, out, params in sweep.calls:
            assert params["SEED"] == 44
            assert params["EXCLUDE_SESSIONS"] == run_mazes.EXCLUDE_SESSIONS[paradigm]
            assert out == run_mazes.bundle_dir(sweep.root, paradigm, 44)
            assert "STEP_EPOCHS" not in params  # full budgets

    def test_a_maze_without_exclusions_gets_an_empty_list(self, sweep):
        """A paradigm with no curated exclusions excludes nothing."""
        sweep("--paradigm", "Nested_Maze", "--seeds", "42")
        assert sweep.calls[0][2]["EXCLUDE_SESSIONS"] == []

    def test_quick_runs_one_seed_on_small_budgets_in_its_own_tree(self, sweep):
        """--quick uses the first seed, cut budgets, and the _quick tree, so a smoke run can
        never be mistaken for (or skip) a real bundle."""
        _complete(
            run_mazes.bundle_dir(sweep.root, "Y_Maze", 42)
        )  # a real bundle exists
        sweep("--paradigm", "Y_Maze", "--quick")
        assert len(sweep.calls) == 1
        _, out, params = sweep.calls[0]
        assert params["SEED"] == run_mazes.DEFAULT_SEEDS[0]
        assert "_quick" in out.parts and params["QUICK_SWEEP"] is True
        assert params["STEP_EPOCHS"] < 10 and params["MLP_EPOCHS"] < 10


def test_exclusions_name_analyzed_mazes_and_real_sessions():
    """Every exclusion key is one of the swept mazes (a typo would exclude nothing) and every
    entry is a well-formed M<id>_<YYMMDD> session name."""
    assert set(run_mazes.EXCLUDE_SESSIONS) <= set(run_mazes.DEFAULT_PARADIGMS)
    for sessions in run_mazes.EXCLUDE_SESSIONS.values():
        assert len(sessions) == len(set(sessions))
        for s in sessions:
            assert re.fullmatch(r"M\d+_\d{6}", s), s
