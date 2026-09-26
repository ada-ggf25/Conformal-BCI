#!/usr/bin/env python
"""Pilot: a session-by-session chain against a fresh model per session, on one maze.

The chain orders every usable session of the maze (all mice) by calendar date. The first
``--warmup`` sessions are split 70/30 (train / early stopping) and never tested. Every later
session is split 60/30/10 (train / calibration and early stopping / test): the encoder, the
alignment decoder and the conformal scale sigma are carried over from the previous session and
fine-tuned on its training block, early-stopped on its calibration block, then tested on its
test block. The KL warm-up runs only for the first session, the one trained from scratch.

The per-region thresholds q_hat are computed several ways from the same scores, so the pool rule
can be chosen from the results:

- ``A``: this session's calibration block only.
- ``B<M>``: this session's block; a region with too few windows borrows the last M sessions'
  calibration blocks (re-scored with today's model); the global threshold otherwise.
- ``C<M>``: this session's block pooled with the last M sessions' blocks, for every region.
- ``D``: this session's block pooled with every past session's block.

Past calibration blocks are re-scored with the current model; test blocks are never pooled.

Baselines, on the same test windows:

- ``fresh``: a new model per tested session, trained from scratch on its 60% and calibrated on
  its own block (rule A).
- ``pooled``: the existing pooled bundle (``analyses/artifacts/<Maze>/ALL_MICE/``), when present.
  Its test block is the same last 10% of each session.

Writes ``analyses/artifacts/pilot_chain/<Maze>/seed_<k>/results.json`` and ``pilot.png``.

Examples:
    uv run python analyses/pilot_chain.py --paradigm Nested_All       # needs the database
    uv run python analyses/pilot_chain.py --paradigm Y_Maze
    uv run python analyses/pilot_chain.py --synthetic --quick         # offline plumbing check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in _HERE.parents if (p / "pyproject.toml").exists()), _HERE.parent
)
for _p in (REPO_ROOT, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# deferred: needs REPO_ROOT on sys.path first
from src.conformal.calibrate import compute_quantile  # noqa: E402
from src.conformal.error_correlation import weighted_pearson  # noqa: E402
from src.conformal.learned import ScaleNet, scale_of, train_scale  # noqa: E402
from src.conformal.predict import class_conditional_prediction_sets  # noqa: E402
from src.conformal.signals import (  # noqa: E402
    SignalStats,
    standardize_signals,
)
from src.conformal.strain import (  # noqa: E402
    compute_encoder_spectral_norms,
    compute_spectral_norms,
)
from src.data.config import WINDOW_SIZE  # noqa: E402
from src.data.datasets import (  # noqa: E402
    HybridClassificationDataset,
    LatentSequenceDataset,
    hybrid_collate_fn,
)
from src.models.alignment import AlignmentDecoder  # noqa: E402
from src.models.config import (  # noqa: E402
    BATCH_SIZE,
    KL_WARMUP_EPOCHS,
    LATENT_DIM,
    STEP1_LEARNING_RATE,
    STEP2_LEARNING_RATE,
    STEP2_SEQ_LEN,
)
from src.models.encoder import SharedTemporalClassifier  # noqa: E402
from src.models.latents import (  # noqa: E402
    build_latent_sequences,
    extract_aligned_latents,
    extract_latents_hybrid,
    extract_logvar,
)
from src.models.training import train_step1, train_step2  # noqa: E402
from src.utils.artifacts import paradigm_slug  # noqa: E402
from src.utils.pipeline_prep import (  # noqa: E402
    compute_class_counts,
    normalize_session_windows,
)
from src.utils.production_config import (  # noqa: E402
    MLP_CAL_SPLIT,
    MLP_DEPTH,
    MLP_HIDDEN,
    MLP_KAPPA,
    MLP_SIZE_WEIGHT,
    MLP_TAU,
    STEP1_CLASS_WEIGHT_POWER,
    STEP1_KL_FREE_BITS,
    STEP1_KL_WEIGHT,
    STEP1_REGION_WEIGHT,
    STEP1_WEIGHT_DECAY,
    STEP2_CLASS_WEIGHT_POWER,
    STEP2_CONTRASTIVE_WEIGHT,
    STEP2_DROPOUT,
)
from src.utils.splits import chronological_task_split  # noqa: E402

ARTIFACTS = REPO_ROOT / "analyses" / "artifacts"
ALPHA = 0.10
FLOOR = (
    int(np.ceil(1 / ALPHA)) - 1
)  # a region keeps its own quantile with > FLOOR windows
RULES = ["A", "B3", "B5", "B10", "C3", "C5", "C10", "D"]
MIN_VAL_WINDOWS, MIN_TEST_WINDOWS = 50, 20


# --------------------------------------------------------------------------- data


def load_maze(paradigm: str) -> tuple[dict, list[str], dict, int, dict[str, str]]:
    """Every usable session of the maze, split 60/30/10, labeled and gated as in per_maze."""
    from run_mazes import EXCLUDE_SESSIONS

    from src.db.client import get_database
    from src.utils.cohort import apply_session_exclusions, discover_cohort
    from src.utils.pipeline_prep import (
        apply_window_region_labels,
        assemble_session_data,
        force_session_specific,
        gate_sessions_conformal,
    )
    from src.utils.production_config import DROP_STRADDLE_WINDOWS, LABEL_REDUCE

    db = get_database()
    cohort = discover_cohort(db, paradigm)
    group = apply_session_exclusions(
        cohort.group, EXCLUDE_SESSIONS.get(paradigm, []), log_fn=print
    )
    session_data, valid, configs, _, n_shared = assemble_session_data(
        db,
        group,
        cohort.label,
        train_frac=0.6,
        val_frac=0.3,
        test_frac=0.1,
    )
    if n_shared > 0:
        force_session_specific(configs)
    apply_window_region_labels(
        session_data,
        valid,
        cohort.label,
        reduce=LABEL_REDUCE,
        drop_straddle=DROP_STRADDLE_WINDOWS,
        log_fn=print,
    )
    dropped = gate_sessions_conformal(
        session_data, valid, MIN_VAL_WINDOWS, MIN_TEST_WINDOWS
    )
    if dropped:
        print(f"gated out {len(dropped)} session(s): {[d[0] for d in dropped]}")
    mouse = {s: str(cohort.session_to_mouse[s]) for s in valid}
    return session_data, valid, configs, cohort.n_classes, mouse


def synthetic_maze(
    n_sessions: int = 6, n_classes: int = 4, seed: int = 0
) -> tuple[dict, list[str], dict, int, dict[str, str]]:
    """Small fake sessions with the same structure, for an offline plumbing check."""
    rng = np.random.default_rng(seed)
    session_data, configs, mouse, names = {}, {}, {}, []
    for i in range(n_sessions):
        name = f"M{1 + i % 2}_2401{10 + i:02d}"
        n_win, n_neur = 400, int(rng.integers(6, 10))
        regions = (np.arange(n_win) // 12) % n_classes  # laps through the regions
        if i % 2:  # the last region only appears at the very end (the test block)
            late = np.arange(n_win) < round(0.9 * n_win)
            regions[late & (regions == n_classes - 1)] = n_classes - 2
        tuning = rng.uniform(0.5, 4.0, size=(n_classes, n_neur))
        counts = rng.poisson(tuning[regions][:, None, :] * np.ones((1, WINDOW_SIZE, 1)))
        tr, va, te = chronological_task_split(np.ones(n_win, bool), 0.6, 0.3, 0.1)
        session_data[name] = {
            "windows": counts.astype("float32"),
            "regions": regions.astype(int),
            "train_idx": tr,
            "val_idx": va,
            "test_idx": te,
        }
        configs[name] = {
            "n_total": n_neur,
            "shared_indices": [],
            "specific_indices": list(range(n_neur)),
            "n_specific": n_neur,
            "n_shared": 0,
        }
        mouse[name] = name.split("_")[0]
        names.append(name)
    return session_data, names, configs, n_classes, mouse


def chain_order(sessions: list[str]) -> list[str]:
    """Calendar order across mice: by recording date (YYMMDD), then mouse."""
    return sorted(sessions, key=lambda s: (s.split("_")[1], s.split("_")[0]))


def resplit_warmup(d: dict) -> None:
    """Warm-up sessions are never tested: fold every task window into a 70/30 split."""
    task = np.sort(np.concatenate([d["train_idx"], d["val_idx"], d["test_idx"]]))
    cut = round(0.7 * len(task))
    d["train_idx"], d["val_idx"], d["test_idx"] = (
        task[:cut],
        task[cut:],
        task[:0],
    )


# --------------------------------------------------------------------------- models


class Pipeline:
    """One encoder + alignment decoder + conformal scale, trained session by session."""

    def __init__(self, configs: dict, n_classes: int, device: torch.device) -> None:
        self.step1 = SharedTemporalClassifier(
            n_shared=0, session_configs=configs, n_classes=n_classes
        ).to(device)
        self.dec = AlignmentDecoder(
            latent_dim=LATENT_DIM,
            n_classes=n_classes,
            seq_len=STEP2_SEQ_LEN,
            dropout=STEP2_DROPOUT,
        ).to(device)
        self.sigma: ScaleNet | None = None
        self.stats: SignalStats | None = None
        self.n_classes, self.device = n_classes, device

    def forward(self, s: str, d: dict) -> dict[str, np.ndarray]:
        """Latents, log-variances and softmax probabilities for every window of a session."""
        self.step1.eval()
        self.dec.eval()
        lat = extract_latents_hybrid(self.step1, d["windows"], s, self.device)
        lv = extract_logvar(self.step1, d["windows"], s, self.device)
        with torch.no_grad():
            al = extract_aligned_latents(self.dec, lat, self.device, STEP2_SEQ_LEN)
            logits = self.dec.decoder(torch.from_numpy(al).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        return {"latents": lat, "logvar": lv, "probs": probs}

    def signals(
        self, s: str, d: dict, fw: dict[str, np.ndarray], idx: np.ndarray
    ) -> np.ndarray:
        """The four uncertainty signals (SIGNALS order) for the windows ``idx``."""
        p = fw["probs"][idx]
        ent = -(p * np.log(np.clip(p, 1e-10, 1.0))).sum(axis=1)
        var = np.exp(fw["logvar"][idx].astype(np.float64)).mean(axis=1)
        seqs = build_latent_sequences(fw["latents"], STEP2_SEQ_LEN)[idx]
        aj = compute_spectral_norms(self.dec, seqs, self.device)
        ej = compute_encoder_spectral_norms(
            self.step1, d["windows"][idx], s, self.device
        )
        return np.column_stack([ent, var, aj, ej]).astype(np.float64)

    def fit(self, s: str, d: dict, *, from_scratch: bool, cfg: argparse.Namespace):
        """Train (or fine-tune) all three stages on this session's training block."""
        batch = min(BATCH_SIZE, 256)
        one = {s: d}
        counts = compute_class_counts(one, [s])
        quiet = cfg.epochs + 1  # log_every beyond the budget: silent
        self.step1, _ = train_step1(
            model=self.step1,
            loader=DataLoader(
                HybridClassificationDataset(one, index_key="train_idx"),
                batch_size=batch,
                shuffle=True,
                collate_fn=hybrid_collate_fn,
            ),
            val_loader=DataLoader(
                HybridClassificationDataset(one, index_key="val_idx"),
                batch_size=batch,
                shuffle=False,
                collate_fn=hybrid_collate_fn,
            ),
            device=self.device,
            log_fn=lambda _m: None,
            n_classes=self.n_classes,
            num_epochs=cfg.epochs,
            lr=STEP1_LEARNING_RATE,
            classifier_weight_decay=STEP1_WEIGHT_DECAY,
            region_weight=STEP1_REGION_WEIGHT,
            class_weight_power=STEP1_CLASS_WEIGHT_POWER,
            class_counts=counts,
            kl_weight=STEP1_KL_WEIGHT,
            kl_free_bits=STEP1_KL_FREE_BITS,
            kl_warmup_epochs=KL_WARMUP_EPOCHS if from_scratch else 0,
            early_stop_patience=cfg.patience,
            log_every=quiet,
        )
        d["step1_all_latents"] = extract_latents_hybrid(
            self.step1, d["windows"], s, self.device
        )
        self.dec, _ = train_step2(
            alignment_decoder=self.dec,
            loader=DataLoader(
                LatentSequenceDataset(one, [s], STEP2_SEQ_LEN, "train_idx"),
                batch_size=batch,
                shuffle=True,
            ),
            val_loader=DataLoader(
                LatentSequenceDataset(one, [s], STEP2_SEQ_LEN, "val_idx"),
                batch_size=batch,
                shuffle=False,
            ),
            device=self.device,
            log_fn=lambda _m: None,
            n_classes=self.n_classes,
            num_epochs=cfg.epochs,
            lr=STEP2_LEARNING_RATE,
            class_counts=counts,
            class_weight_power=STEP2_CLASS_WEIGHT_POWER,
            contrastive_weight=STEP2_CONTRASTIVE_WEIGHT,
            early_stop_patience=cfg.patience,
            log_every=quiet,
        )
        fw = self.forward(s, d)
        tr, va = d["train_idx"], d["val_idx"]
        u_tr, u_va = self.signals(s, d, fw, tr), self.signals(s, d, fw, va)
        self.stats = SignalStats.fit(u_tr)
        y = d["regions"]
        self.sigma, _ = train_scale(
            standardize_signals(u_tr, self.stats),
            fw["probs"][tr],
            y[tr],
            alpha=ALPHA,
            hidden=MLP_HIDDEN,
            depth=MLP_DEPTH,
            epochs=cfg.mlp_epochs,
            tau=MLP_TAU,
            kappa=MLP_KAPPA,
            size_weight=MLP_SIZE_WEIGHT,
            val_signals_z=standardize_signals(u_va, self.stats),
            val_probs=fw["probs"][va],
            val_labels=y[va],
            early_stop_patience=cfg.mlp_patience,
            n_classes=self.n_classes,
            split=MLP_CAL_SPLIT,
            groups=np.zeros(len(tr), int),
            val_groups=np.zeros(len(va), int),
            init_net=None if from_scratch else self.sigma,
            seed=cfg.seed,
            device=str(self.device),
        )

    def score(self, s: str, d: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Probabilities, sigma and true-class scores for ``idx`` under the current model."""
        assert self.sigma is not None and self.stats is not None
        fw = self.forward(s, d)
        u = standardize_signals(self.signals(s, d, fw, idx), self.stats)
        sig = scale_of(self.sigma, u, device=str(self.device))
        probs, y = fw["probs"][idx], d["regions"][idx]
        return {
            "probs": probs,
            "sigma": sig,
            "labels": y,
            "scores": (1.0 - probs[np.arange(len(idx)), y]) / sig,
        }


# --------------------------------------------------------------------------- calibration


def qhats_for_rule(
    rule: str, own: dict, past: list[dict], n_classes: int
) -> tuple[dict[int, float], dict[int, str]]:
    """Per-region thresholds and where each came from (own / borrowed / global)."""
    m = None if rule in ("A", "D") else int(rule[1:])
    sel = [] if rule == "A" else (past if rule == "D" else past[-m:])
    pool_s = np.concatenate([own["scores"], *[p["scores"] for p in sel]])
    pool_y = np.concatenate([own["labels"], *[p["labels"] for p in sel]])
    own_global = compute_quantile(own["scores"], ALPHA)
    pool_global = compute_quantile(pool_s, ALPHA)
    q, level = {}, {}
    for c in range(n_classes):
        n_own = int((own["labels"] == c).sum())
        n_pool = int((pool_y == c).sum())
        if rule == "A" or (rule.startswith("B") and n_own > FLOOR):
            use_own = n_own > FLOOR
            q[c] = (
                compute_quantile(own["scores"][own["labels"] == c], ALPHA)
                if use_own
                else own_global
            )
            level[c] = "own" if use_own else "global"
        elif rule.startswith("B"):
            if n_pool > FLOOR:
                q[c], level[c] = (
                    compute_quantile(pool_s[pool_y == c], ALPHA),
                    "borrowed",
                )
            else:
                q[c], level[c] = own_global, "global"
        elif n_pool > FLOOR:  # C, D: the pooled block for every region
            q[c] = compute_quantile(pool_s[pool_y == c], ALPHA)
            level[c] = "own" if n_own > FLOOR else "borrowed"
        else:
            q[c], level[c] = pool_global, "global"
    return q, level


def evaluate(test: dict, q: dict[int, float], level: dict[int, str], k: int) -> dict:
    """Per-window coverage, size, error and threshold level on a test block."""
    sets = class_conditional_prediction_sets(test["probs"], test["sigma"], q)
    y = test["labels"]
    return {
        "covered": np.array([y[i] in s for i, s in enumerate(sets)], float),
        "size_over_k": np.array([len(s) for s in sets], float) / k,
        "error": (test["probs"].argmax(1) != y).astype(float),
        "sigma_rel": test["sigma"] / np.median(test["sigma"]),
        "level": np.array([level[int(c)] for c in y]),
    }


# --------------------------------------------------------------------------- reporting


def grouped_corr(x: np.ndarray, err: np.ndarray, *, bins: int | None) -> float:
    """Count-weighted Pearson of error rate over groups of x (levels, or equal-count bins)."""
    if bins is None:
        groups = [np.flatnonzero(x == v) for v in np.unique(x)]
    else:
        groups = np.array_split(np.argsort(x, kind="stable"), bins)
    groups = [g for g in groups if len(g)]
    return weighted_pearson(
        np.array([x[g].mean() for g in groups]),
        np.array([err[g].mean() for g in groups]),
        np.array([len(g) for g in groups]),
    )


def summarize(rows: list[dict], key: str) -> dict:
    """Aggregate one method (per-session rows) over the tested sessions."""
    ev = [r[key] for r in rows]
    cat = {f: np.concatenate([e[f] for e in ev]) for f in ev[0]}
    cov_s = np.array([e["covered"].mean() for e in ev])
    return {
        "sessions": len(ev),
        "coverage": float(cat["covered"].mean()),
        "mean_abs_coverage_gap": float(np.mean(np.abs(cov_s - (1 - ALPHA)))),
        "sessions_below_target": int((cov_s < 1 - ALPHA).sum()),
        "size_over_k": float(cat["size_over_k"].mean()),
        "accuracy": float(1 - cat["error"].mean()),
        "share_own": float((cat["level"] == "own").mean()),
        "share_borrowed": float((cat["level"] == "borrowed").mean()),
        "share_global": float((cat["level"] == "global").mean()),
        "corr_error_size": grouped_corr(cat["size_over_k"], cat["error"], bins=None),
        "corr_error_sigma_rel": grouped_corr(cat["sigma_rel"], cat["error"], bins=10),
    }


def pooled_reference(slug: str, tested: list[str]) -> dict[str, dict] | None:
    """Per-session results of the existing pooled bundle on the same sessions, if present."""
    b = ARTIFACTS / slug / "ALL_MICE"
    if not (b / "arrays.npz").exists():
        return None
    met = json.loads((b / "metrics.json").read_text())
    a = np.load(b / "arrays.npz")
    q_raw = met["conformal"]["arms"]["learned-scale class-conditional"][
        "q_hat_by_region"
    ]
    q = {int(c): (np.inf if v is None else float(v)) for c, v in q_raw.items()}
    out = {}
    for si, s in enumerate(met["sessions"]):
        if s not in tested:
            continue
        m = (a["win_session_idx"] == si) & (a["win_split"] == 2)
        test = {
            "probs": a["win_probs"][m].astype(float),
            "sigma": a["win_sigma"][m].astype(float),
            "labels": a["win_region"][m],
        }
        out[s] = evaluate(test, q, dict.fromkeys(q, "own"), met["n_classes"])
    return out


def plot(chain_rows: list[dict], ref: dict | None, rule: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(chain_rows))
    first = [i for i, r in enumerate(chain_rows) if r["first_of_mouse"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a_, f, label in ((ax[0], "error", "accuracy"), (ax[1], "covered", "coverage")):
        tf = (lambda v: 1 - v) if f == "error" else (lambda v: v)
        a_.plot(
            x,
            [tf(r[f"chain_{rule}"][f].mean()) for r in chain_rows],
            "o-",
            label=f"chain ({rule})",
        )
        a_.plot(
            x,
            [tf(r["fresh"][f].mean()) for r in chain_rows],
            "s-",
            label="fresh per session",
        )
        if ref:
            a_.plot(
                x,
                [
                    tf(ref[r["session"]][f].mean()) if r["session"] in ref else np.nan
                    for r in chain_rows
                ],
                "^-",
                label="pooled (thesis)",
            )
        for i in first:
            a_.axvline(i, color="0.8", lw=1, zorder=0)
        a_.set_xlabel(
            "tested session, calendar order (grey lines: a mouse's first session)"
        )
        a_.set_ylabel(label)
        a_.grid(alpha=0.3)
    ax[1].axhline(1 - ALPHA, ls="--", color="k", lw=1)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def run(
    session_data: dict,
    sessions: list[str],
    configs: dict,
    n_classes: int,
    mouse: dict[str, str],
    cfg: argparse.Namespace,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    chain = chain_order(sessions)
    if len(chain) <= cfg.warmup:
        raise SystemExit(
            f"{len(chain)} sessions: nothing left to test after the warm-up"
        )
    for s in chain[: cfg.warmup]:
        resplit_warmup(session_data[s])
    normalize_session_windows(session_data, chain)
    print(
        f"chain: {len(chain)} sessions ({len(set(mouse.values()))} mice), "
        f"{cfg.warmup} warm-up, {len(chain) - cfg.warmup} tested"
    )

    pipe = Pipeline(configs, n_classes, device)
    seen: set[str] = set()
    rows: list[dict] = []
    t_all = time.time()
    for i, s in enumerate(chain):
        t0 = time.time()
        d = session_data[s]
        pipe.fit(s, d, from_scratch=(i == 0), cfg=cfg)
        first = mouse[s] not in seen
        seen.add(mouse[s])
        if i < cfg.warmup:
            print(f"[{i + 1}/{len(chain)}] {s}  warm-up  ({time.time() - t0:.0f}s)")
            continue

        own = pipe.score(s, d, d["val_idx"])
        test = pipe.score(s, d, d["test_idx"])
        past_sessions = (
            chain[:i] if cfg.max_past is None else chain[max(0, i - cfg.max_past) : i]
        )
        past = [
            pipe.score(p, session_data[p], session_data[p]["val_idx"])
            for p in past_sessions
        ]
        row: dict[str, Any] = {
            "session": s,
            "mouse": mouse[s],
            "first_of_mouse": first,
            "n_neurons": int(configs[s]["n_total"]),
            "n_cal": len(d["val_idx"]),
            "n_test": len(d["test_idx"]),
        }
        for rule in RULES:
            q, level = qhats_for_rule(rule, own, past, n_classes)
            row[f"chain_{rule}"] = evaluate(test, q, level, n_classes)

        fresh = Pipeline({s: configs[s]}, n_classes, device)
        fresh.fit(s, d, from_scratch=True, cfg=cfg)
        f_own, f_test = (
            fresh.score(s, d, d["val_idx"]),
            fresh.score(s, d, d["test_idx"]),
        )
        q, level = qhats_for_rule("A", f_own, [], n_classes)
        row["fresh"] = evaluate(f_test, q, level, n_classes)
        rows.append(row)

        c5 = row["chain_C5"]
        print(
            f"[{i + 1}/{len(chain)}] {s}{'  (new mouse)' if first else ''}  "
            f"chain acc {1 - c5['error'].mean():.2f} cov {c5['covered'].mean():.2f} "
            f"|C|/K {c5['size_over_k'].mean():.2f}  |  fresh acc "
            f"{1 - row['fresh']['error'].mean():.2f} cov {row['fresh']['covered'].mean():.2f}"
            f"  ({time.time() - t0:.0f}s)"
        )

    slug = paradigm_slug(cfg.paradigm)
    ref = (
        None if cfg.synthetic else pooled_reference(slug, [r["session"] for r in rows])
    )
    groups = {
        "all tested": rows,
        "returning mouse": [r for r in rows if not r["first_of_mouse"]],
        "first session of a mouse": [r for r in rows if r["first_of_mouse"]],
    }
    if ref:  # the pooled bundle may hold fewer sessions: compare every method on the shared ones
        groups["sessions in the pooled bundle"] = [
            r for r in rows if r["session"] in ref
        ]
    summary: dict[str, dict] = {}
    for gname, grp in groups.items():
        if not grp:
            continue
        summary[gname] = {f"chain_{r}": summarize(grp, f"chain_{r}") for r in RULES}
        summary[gname]["fresh"] = summarize(grp, "fresh")
        if ref and all(r["session"] in ref for r in grp):
            for r in grp:
                r["pooled"] = ref[r["session"]] | {
                    "level": np.full(len(ref[r["session"]]["error"]), "own")
                }
            summary[gname]["pooled (thesis)"] = summarize(grp, "pooled")

    print(f"\ndone in {(time.time() - t_all) / 60:.1f} min")
    for gname, methods in summary.items():
        print(f"\n== {gname} ({next(iter(methods.values()))['sessions']} sessions)")
        print(
            f"  {'method':<16} {'cov':>6} {'|gap|':>6} {'below':>6} {'|C|/K':>6} "
            f"{'acc':>6} {'own':>5} {'borr':>5} {'glob':>5} {'r(size)':>8} {'r(sig)':>7}"
        )
        for name, m in methods.items():
            print(
                f"  {name:<16} {m['coverage']:>6.3f} {m['mean_abs_coverage_gap']:>6.3f} "
                f"{m['sessions_below_target']:>6d} {m['size_over_k']:>6.3f} {m['accuracy']:>6.3f} "
                f"{m['share_own']:>5.2f} {m['share_borrowed']:>5.2f} {m['share_global']:>5.2f} "
                f"{m['corr_error_size']:>+8.2f} {m['corr_error_sigma_rel']:>+7.2f}"
            )

    out_dir = (
        ARTIFACTS / "pilot_chain" / ("_synthetic" if cfg.synthetic else slug)
    ) / f"seed_{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_session = [
        {k: v for k, v in r.items() if not isinstance(v, dict)}
        | {
            m: {
                "accuracy": float(1 - r[m]["error"].mean()),
                "coverage": float(r[m]["covered"].mean()),
                "size_over_k": float(r[m]["size_over_k"].mean()),
                "share_global": float((r[m]["level"] == "global").mean()),
            }
            for m in r
            if isinstance(r[m], dict)
        }
        for r in rows
    ]
    results = {
        "config": {k: v for k, v in vars(cfg).items()},
        "chain": chain,
        "per_session": per_session,
        "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot(rows, ref, "C5", out_dir / "pilot.png")
    print(f"\nwrote {out_dir / 'results.json'} and pilot.png")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paradigm", default="Nested_All")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--epochs", type=int, default=200, help="Step 1 and Step 2 epoch cap"
    )
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--mlp-epochs", type=int, default=300)
    ap.add_argument("--mlp-patience", type=int, default=30)
    ap.add_argument(
        "--max-past",
        type=int,
        default=None,
        help="re-score at most this many past sessions (rule D uses all of them); default all",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="2-epoch smoke run")
    ap.add_argument(
        "--synthetic", action="store_true", help="fake sessions, no database"
    )
    cfg = ap.parse_args()
    if cfg.quick:
        cfg.epochs, cfg.patience, cfg.mlp_epochs, cfg.mlp_patience = 2, 2, 5, 3
    data = synthetic_maze(seed=cfg.seed) if cfg.synthetic else load_maze(cfg.paradigm)
    run(*data, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
