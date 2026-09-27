"""Tests for src/chain/runner.py: end-to-end on small synthetic sessions.

Training budgets are cut to a single epoch, so the numbers mean nothing; the tests check
properties of the protocol that must hold whatever the model learns.

Accepted gaps: the quality of what is learned (real-data validation covers it).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import torch

from src.chain.order import prepare_chain
from src.chain.runner import SWEEP_TARGETS, ChainConfig, _save, _unit_scores, run_chain
from src.chain.stages import TrainBudget
from src.data.config import WINDOW_SIZE
from src.utils.splits import chronological_task_split

N_CLASSES = 4
CFG = ChainConfig(
    seed=0,
    warmup=2,
    sensitivity=("B1",),
    sweep_targets=(0.5, 0.9),
    budget=TrainBudget(epochs=1, patience=1, mlp_epochs=2, mlp_patience=1),
)


def _sessions(n):
    """n small sessions of two mice; odd sessions show the last region only in the test block."""
    rng = np.random.default_rng(0)
    data, configs, names = {}, {}, []
    for i in range(n):
        name = f"M{1 + i % 2}_2401{10 + i:02d}"
        n_win, n_neur = 80, int(rng.integers(5, 8))
        regions = (np.arange(n_win) // 5) % N_CLASSES
        if i % 2:
            late = np.arange(n_win) < round(0.9 * n_win)
            regions[late & (regions == N_CLASSES - 1)] = N_CLASSES - 2
        tuning = rng.uniform(0.5, 4.0, size=(N_CLASSES, n_neur))
        counts = rng.poisson(tuning[regions][:, None, :] * np.ones((1, WINDOW_SIZE, 1)))
        tr, va, te = chronological_task_split(np.ones(n_win, bool), 0.6, 0.3, 0.1)
        data[name] = {
            "windows": counts.astype("float32"),
            "regions": regions,
            "train_idx": tr,
            "val_idx": va,
            "test_idx": te,
        }
        configs[name] = {
            "n_total": n_neur,
            "shared_indices": [],
            "specific_indices": list(range(n_neur)),
            "n_specific": n_neur,
        }
        names.append(name)
    return data, names, configs


def _run(n, cfg=CFG, log=lambda _m: None, **kwargs):
    """Prepare n synthetic sessions and run the chain over them."""
    data, names, configs = _sessions(n)
    chain = prepare_chain(data, names, cfg.warmup)
    mouse = {s: s.split("_")[0] for s in chain}
    return run_chain(data, chain, configs, N_CLASSES, mouse, cfg, log=log, **kwargs)


@cache
def _uninterrupted():
    """The reference five-session run, shared by the tests."""
    return _run(5)


def _same(a, b):
    """Two runs' rows hold identical results for every method and sweep target."""
    assert [r["session"] for r in a] == [r["session"] for r in b]
    for ra, rb in zip(a, b, strict=True):
        for m in ra["methods"]:
            for k, v in ra["methods"][m].items():
                np.testing.assert_array_equal(
                    v, rb["methods"][m][k], err_msg=f"{m}.{k}"
                )
        for t in ra["sweep"]:
            np.testing.assert_array_equal(
                ra["sweep"][t]["covered"], rb["sweep"][t]["covered"]
            )


def test_later_sessions_never_change_earlier_results():
    """Causality: a chain extended by one more session reproduces every earlier session exactly.
    Also: warm-up sessions are not tested, and the sweep's 90% point is the deployed result."""
    short, full = _run(4), _uninterrupted()
    assert len(short) == 2 and len(full) == 3
    _same(short, full[:2])
    r = full[0]
    np.testing.assert_array_equal(
        r["sweep"][0.9]["covered"], r["methods"]["chain_B3"]["covered"]
    )


def test_resuming_from_a_checkpoint_reproduces_the_uninterrupted_run(tmp_path):
    """A chain stopped after 3 sessions and resumed gives the same results as one run through."""
    ckpt = tmp_path / "chain.pt"
    _run(5, checkpoint=ckpt, stop_after=3)
    _same(_run(5, checkpoint=ckpt), _uninterrupted())


def test_rows_record_the_tested_sessions_test_block():
    """Each row's labels are its test windows' regions, and sigma_rel is relative to that
    session's median (so its median is exactly 1)."""
    data, _, _ = _sessions(5)
    rows = _uninterrupted()
    for r in rows:
        d = data[r["session"]]
        assert r["labels"].tolist() == d["regions"][d["test_idx"]].tolist()
        assert r["n_test"] == len(d["test_idx"]) and r["n_cal"] == len(d["val_idx"])
        assert np.median(r["sigma_rel"]) == pytest.approx(1.0)
        assert r["mouse"] == r["session"].split("_")[0]
    assert [r["first_of_mouse"] for r in rows] == [False, False, False]


def test_methods_sweep_and_baseline_switch():
    """The deployed rule, its sensitivity rules and the sigma = 1 ablation are always run;
    the fresh baseline only when asked. The sweep has one entry per configured target."""
    logs: list[str] = []
    cfg = ChainConfig(
        seed=0, warmup=1, sensitivity=("A",), sweep_targets=(0.9,), baseline=False,
        budget=CFG.budget,
    )  # fmt: skip
    rows = _run(2, cfg=cfg, log=logs.append)
    assert set(rows[0]["methods"]) == {"chain_B3", "chain_A", "chain_sigma1"}
    assert set(_uninterrupted()[0]["methods"]) == {
        "chain_B3", "chain_B1", "chain_sigma1", "fresh",
    }  # fmt: skip
    assert list(rows[0]["sweep"]) == [0.9]
    assert len(logs) == 2 and "warm-up" in logs[0] and "(new mouse)" in logs[1]


def test_unit_scores_are_one_minus_the_true_probability():
    """The sigma = 1 ablation scores 1 - p_y with a unit scale."""
    blk = {"probs": np.array([[0.1, 0.9], [0.6, 0.4]]), "labels": np.array([1, 1])}
    out = _unit_scores(blk)
    np.testing.assert_allclose(out["scores"], [0.1, 0.6])
    assert out["sigma"].tolist() == [1.0, 1.0]


def test_sweep_grid_includes_the_deployed_target():
    """The sweep runs 5% to 95% in 5% steps, so the deployed 90% is one of its points."""
    assert len(SWEEP_TARGETS) == 19
    assert SWEEP_TARGETS[0] == pytest.approx(0.05) and SWEEP_TARGETS[
        -1
    ] == pytest.approx(0.95)
    assert any(np.isclose(t, 0.9) for t in SWEEP_TARGETS)


def test_checkpoint_write_is_atomic(tmp_path):
    """The checkpoint lands under its final name (in a created directory) with no temp file."""

    class _Model:
        def state(self):
            """A stand-in model state."""
            return {"w": 1}

    path = tmp_path / "sub" / "chain.pt"
    _save(path, 3, _Model(), [{"session": "S"}])
    assert path.exists() and not path.with_suffix(".tmp").exists()
    state = torch.load(path, weights_only=False)
    assert (
        state["next"] == 3
        and state["model"] == {"w": 1}
        and state["rows"][0]["session"] == "S"
    )
