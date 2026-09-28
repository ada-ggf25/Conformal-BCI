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

from src.chain.evaluate import evaluate_block
from src.chain.order import prepare_chain
from src.chain.pool import pooled_qhats
from src.chain.runner import SWEEP_TARGETS, ChainConfig, _save, _unit_scores, run_chain
from src.chain.stages import TrainBudget
from src.conformal.learned import ScaleNet, scale_of
from src.conformal.signals import SIGNALS, SignalStats, standardize_signals
from src.data.config import WINDOW_SIZE
from src.models.config import MAIN_EARLY_STOP_MIN_DELTA
from src.utils.production_config import MLP_DEPTH, MLP_HIDDEN
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
        """Stand-in for ChainModel: _save only reads its state()."""

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


def test_starved_regions_borrow_from_past_sessions():
    """Odd sessions see their last region only in the test block, so it is starved in their
    own calibration block and must borrow: some test windows are labeled "borrowed"."""
    levels = np.concatenate(
        [r["methods"]["chain_B3"]["level"] for r in _uninterrupted()]
    )
    assert "borrowed" in set(levels.tolist())


def test_deeper_borrowing_never_falls_back_more_often():
    """B3 pools a superset of B1's past, so wherever B1 had enough windows B3 does too: B3 has
    no more global fallbacks than B1, and the regions with their own threshold are the same."""
    for r in _uninterrupted():
        b3, b1 = r["methods"]["chain_B3"]["level"], r["methods"]["chain_B1"]["level"]
        assert (b3 == "global").sum() <= (b1 == "global").sum()
        np.testing.assert_array_equal(b3 == "own", b1 == "own")


def test_the_fresh_baseline_is_trained_from_scratch(monkeypatch):
    """The baseline is a new model per session with standard conformal: fitted from scratch
    (with the KL warm-up) and without a sigma."""
    from src.chain import runner

    calls: list[dict] = []
    real_fit = runner.ChainModel.fit

    def spy(self, *args, **kwargs):
        """Record how each fit is called, then fit for real."""
        calls.append(kwargs)
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(runner.ChainModel, "fit", spy)
    cfg = ChainConfig(seed=0, warmup=1, sensitivity=(), sweep_targets=(0.9,),
                      budget=CFG.budget)  # fmt: skip
    _run(2, cfg=cfg)
    baseline = [c for c in calls if c.get("train_sigma") is False]
    assert len(baseline) == 1 and baseline[0]["from_scratch"] is True
    chain_fits = [c for c in calls if c.get("train_sigma", True)]
    assert [c["from_scratch"] for c in chain_fits] == [True, False]


def test_unit_scores_match_a_model_without_sigma():
    """The sigma = 1 ablation scores exactly like a model that has no sigma at all."""
    from src.chain.stages import ChainModel

    blk = {"probs": np.array([[0.3, 0.7], [0.9, 0.1]]), "labels": np.array([0, 1]),
           "u": np.zeros((2, 4))}  # fmt: skip
    m = ChainModel("S", {"n_total": 2, "shared_indices": [], "specific_indices": [0, 1],
                         "n_specific": 2}, 2, torch.device("cpu"))  # fmt: skip
    np.testing.assert_array_equal(_unit_scores(blk)["scores"], m.score(blk)["scores"])


def test_a_new_checkpoint_replaces_the_old_one(tmp_path):
    """Saving again overwrites the previous checkpoint with the newer position."""

    class _Model:
        """Stand-in for ChainModel: _save only reads its state()."""

        def state(self):
            """A stand-in model state."""
            return {}

    path = tmp_path / "chain.pt"
    _save(path, 1, _Model(), [])
    _save(path, 2, _Model(), [{"session": "S"}])
    state = torch.load(path, weights_only=False)
    assert state["next"] == 2 and len(state["rows"]) == 1


def test_chain_config_is_immutable():
    """A run's configuration cannot be changed once built (it is recorded in the bundle)."""
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        CFG.seed = 1  # type: ignore[misc]


def _recalibrate(row, rule, unit=False):
    """Rebuild a method's test results from a row's stored scores alone (the offline path)."""
    cal = sorted(row["calibration"], key=lambda c: c["offset"])

    def block(c):
        scores = c["scores"] * c["sigma"] if unit else c["scores"]
        return {"labels": c["labels"], "scores": scores}

    own, past = block(cal[0]), [block(c) for c in cal[1:]][::-1]  # past: oldest first
    q, levels = pooled_qhats(rule, own, past, N_CLASSES, CFG.alpha)
    sigma = np.ones_like(row["sigma"]) if unit else row["sigma"]
    return evaluate_block(
        {"probs": row["probs"], "sigma": sigma, "labels": row["labels"]},
        q,
        levels,
        N_CLASSES,
    )


def test_stored_scores_reproduce_every_calibrated_method():
    """Offline re-calibration from a row's probabilities, sigma and calibration scores gives
    exactly the chain's own results, for the method and for the sigma = 1 ablation."""
    for row in _uninterrupted():
        assert [c["offset"] for c in row["calibration"]] == list(
            range(len(row["calibration"]))
        )
        for rule, unit, method in (("B3", False, "chain_B3"), ("B1", False, "chain_B1"),
                                   ("B3", True, "chain_sigma1")):  # fmt: skip
            got = _recalibrate(row, rule, unit)
            for k in ("covered", "size_over_k", "error", "level"):
                np.testing.assert_array_equal(
                    got[k], row["methods"][method][k], err_msg=method
                )


@cache
def _logged():
    """The reference run with its training log."""
    log: list = []
    rows = _run(5, training_log=log)
    return rows, log


def test_training_log_covers_every_session_with_consistent_curves():
    """One entry per chain session, warm-up flagged; each stage's curve has one loss per
    epoch, and no epoch after the restored one beat it by more than the stopping tolerance."""
    rows, log = _logged()
    assert [e["warmup"] for e in log] == [True, True, False, False, False]
    assert [e["session"] for e in log][CFG.warmup :] == [r["session"] for r in rows]
    for e in log:
        for stage in ("step1", "step2", "sigma"):
            c = e[stage]
            assert len(c["train_loss"]) == len(c["val_loss"]) == c["epochs"] >= 1
            assert c["best_epoch"] is None or 1 <= c["best_epoch"] <= c["epochs"]
            if c["best_epoch"] is not None:
                restored = c["val_loss"][c["best_epoch"] - 1]
                assert min(c["val_loss"][c["best_epoch"] :], default=np.inf) >= (
                    restored - MAIN_EARLY_STOP_MIN_DELTA
                )


def test_a_resumed_run_keeps_the_full_training_log(tmp_path):
    """A chain stopped after 3 sessions and resumed ends with the uninterrupted run's log."""
    ckpt = tmp_path / "chain.pt"
    _run(5, checkpoint=ckpt, stop_after=3, training_log=[])
    resumed: list = []
    _run(5, checkpoint=ckpt, training_log=resumed)
    assert [(e["session"], e["step1"]["train_loss"]) for e in resumed] == [
        (e["session"], e["step1"]["train_loss"]) for e in _logged()[1]
    ]


def test_logged_sigma_inputs_rescore_the_stored_calibration_block():
    """The stored sigma net, applied to the logged calibration signals and probabilities,
    gives exactly the stored own-block sigma and scores (1 - p_y) / sigma: the logged fit
    inputs are the calibration windows, in order, under the tested session's model."""
    rows, log = _logged()
    for j, row in enumerate(rows):
        va = log[CFG.warmup + j]["fit_inputs"]["va"]
        net = ScaleNet(len(SIGNALS), MLP_HIDDEN, MLP_DEPTH)
        net.load_state_dict(row["sigma_state"])
        stats = SignalStats(*(np.asarray(v) for v in row["signal_stats"]))
        sigma = scale_of(net, standardize_signals(va["u"], stats))
        p_true = va["probs"][np.arange(len(va["labels"])), va["labels"]]
        own = row["calibration"][0]
        np.testing.assert_array_equal(own["labels"], va["labels"])
        np.testing.assert_allclose(own["sigma"], sigma, rtol=1e-6)
        np.testing.assert_allclose(own["scores"], (1 - p_true) / sigma, rtol=1e-6)
