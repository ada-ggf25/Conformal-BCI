"""Run a maze's session chain end to end, with a checkpoint after every session."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .evaluate import evaluate_block
from .order import first_session_flags
from .pool import borrow_depth, marginal_qhats, pooled_qhats
from .stages import ChainModel, TrainBudget

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

SWEEP_TARGETS: tuple[float, ...] = tuple(
    float(t) for t in np.round(np.arange(0.05, 0.951, 0.05), 2)
)
"""Coverage targets of the sweep: 5% to 95% in 5% steps (the deployed 90% is one of them)."""


@dataclass(frozen=True)
class ChainConfig:
    """Everything that parameterizes one chain run."""

    seed: int = 42
    device: str = "cpu"
    warmup: int = 3
    alpha: float = 0.10
    rule: str = "B3"
    sensitivity: tuple[str, ...] = ("B1", "B5")
    sweep_targets: tuple[float, ...] = SWEEP_TARGETS
    baseline: bool = True
    budget: TrainBudget = field(default_factory=TrainBudget)


def run_chain(
    session_data: dict,
    chain: list[str],
    configs: dict,
    n_classes: int,
    mouse: dict[str, str],
    cfg: ChainConfig,
    *,
    checkpoint: Path | None = None,
    log: Callable[[str], None] = print,
    stop_after: int | None = None,
) -> list[dict[str, Any]]:
    """Train the chain session by session and evaluate every session after the warm-up.

    Each session: the carried model gets the session's input layer, is fine-tuned on its
    training block (from scratch for the first session), and, after the warm-up, is evaluated
    on its test block: the method (``cfg.rule``), the pool-rule sensitivity, the sigma = 1
    ablation, the coverage sweep and, if ``cfg.baseline``, a fresh model per session with
    standard split conformal. A session is only ever evaluated with the model right after its
    own fine-tune.

    Args:
        session_data: prepared sessions (``prepare_chain``), keyed by name.
        chain:        session names in chain order.
        configs:      per-session neuron configs.
        n_classes:    number of regions.
        mouse:        session -> mouse id.
        cfg:          the run's configuration.
        checkpoint:   file to save after every session and resume from if it exists.
        log:          progress sink.
        stop_after:   stop once this many sessions are done (for interrupting in tests).

    Returns:
        One row per tested session (see ``_evaluate_session``).
    """
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = ChainModel(chain[0], configs[chain[0]], n_classes, device)
    rows: list[dict[str, Any]] = []
    start = 0
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load(state["model"], configs)
        rows, start = state["rows"], state["next"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["np_rng"])
        log(f"resumed at session {start + 1}/{len(chain)}")

    first = first_session_flags(chain, mouse)
    max_past = max(borrow_depth(r) for r in (cfg.rule, *cfg.sensitivity))
    for i in range(start, len(chain)):
        s = chain[i]
        model.add_session(s, configs[s])
        carried_sigma = copy.deepcopy(model.sigma)
        inputs = model.fit(
            s,
            session_data[s],
            from_scratch=(i == 0),
            budget=cfg.budget,
            alpha=cfg.alpha,
            seed=cfg.seed,
        )
        if i >= cfg.warmup:
            past = chain[max(0, i - max_past) : i]
            row = _evaluate_session(
                model, carried_sigma, inputs, s, past, session_data, configs,
                n_classes, cfg, first[i], mouse[s],
            )  # fmt: skip
            rows.append(row)
            b = row["methods"][f"chain_{cfg.rule}"]
            log(
                f"[{i + 1}/{len(chain)}] {s}{'  (new mouse)' if first[i] else ''}  "
                f"acc {1 - b['error'].mean():.2f}  cov {b['covered'].mean():.2f}  "
                f"|C|/K {b['size_over_k'].mean():.2f}"
            )
        else:
            log(f"[{i + 1}/{len(chain)}] {s}  warm-up")
        if checkpoint is not None:
            _save(checkpoint, i + 1, model, rows)
        if stop_after is not None and i + 1 >= stop_after:
            break
    return rows


def _evaluate_session(
    model: ChainModel,
    carried_sigma: Any,
    inputs: dict,
    s: str,
    past_sessions: list[str],
    session_data: dict,
    configs: dict,
    n_classes: int,
    cfg: ChainConfig,
    first_of_mouse: bool,
    mouse_id: str,
) -> dict[str, Any]:
    """Every result for one tested session, from the model right after its own fine-tune."""
    d = session_data[s]
    own_blk = model.block(s, d, d["val_idx"])
    test_blk = model.block(s, d, d["test_idx"])
    past_blks = [
        model.block(p, session_data[p], session_data[p]["val_idx"])
        for p in past_sessions
    ]

    def calibrate(sigma: Any, rule: str, alpha: float, unit: bool = False) -> dict:
        score = (
            (lambda b: _unit_scores(b)) if unit else (lambda b: model.score(b, sigma))
        )
        own, test = score(own_blk), score(test_blk)
        past = [score(b) for b in past_blks]
        q, levels = pooled_qhats(rule, own, past, n_classes, alpha)
        return evaluate_block(test, q, levels, n_classes)

    methods = {
        f"chain_{r}": calibrate(None, r, cfg.alpha)
        for r in (cfg.rule, *cfg.sensitivity)
    }
    methods["chain_sigma1"] = calibrate(None, cfg.rule, cfg.alpha, unit=True)

    sweep: dict[float, dict[str, np.ndarray]] = {}
    for target in cfg.sweep_targets:
        if np.isclose(target, 1 - cfg.alpha):
            ev = methods[
                f"chain_{cfg.rule}"
            ]  # the deployed model is the sweep's 90% point
        else:
            branch = model.train_sigma(
                inputs, 1 - target, carried_sigma, cfg.budget, cfg.seed
            )
            ev = calibrate(branch, cfg.rule, 1 - target)
        sweep[target] = {"covered": ev["covered"], "size_over_k": ev["size_over_k"]}

    if cfg.baseline:
        fresh = ChainModel(s, configs[s], n_classes, model.device)
        fresh.fit(
            s, d, from_scratch=True, budget=cfg.budget, alpha=cfg.alpha,
            seed=cfg.seed, train_sigma=False,
        )  # fmt: skip
        f_own = fresh.score(fresh.block(s, d, d["val_idx"]))
        f_test = fresh.score(fresh.block(s, d, d["test_idx"]))
        q, levels = marginal_qhats(f_own, n_classes, cfg.alpha)
        methods["fresh"] = evaluate_block(f_test, q, levels, n_classes)

    deployed = model.score(test_blk)
    return {
        "session": s,
        "mouse": mouse_id,
        "first_of_mouse": first_of_mouse,
        "n_neurons": int(configs[s]["n_total"]),
        "n_cal": len(d["val_idx"]),
        "n_test": len(d["test_idx"]),
        "labels": test_blk["labels"],
        "sigma_rel": deployed["sigma"] / np.median(deployed["sigma"]),
        "signals": test_blk["u"],
        "methods": methods,
        "sweep": sweep,
        "sigma_state": model.state()["sigma"],
        "signal_stats": model.state()["stats"],
    }


def _unit_scores(blk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scores with sigma = 1 (the ablation): ``1 - p_y``."""
    p_true = blk["probs"][np.arange(len(blk["labels"])), blk["labels"]]
    return {
        "probs": blk["probs"],
        "sigma": np.ones(len(blk["labels"])),
        "labels": blk["labels"],
        "scores": 1.0 - p_true,
    }


def _save(path: Path, next_index: int, model: ChainModel, rows: list) -> None:
    """Write the checkpoint atomically (a crash mid-write keeps the previous one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "next": next_index,
            "model": model.state(),
            "rows": rows,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
        },
        tmp,
    )
    os.replace(tmp, path)
