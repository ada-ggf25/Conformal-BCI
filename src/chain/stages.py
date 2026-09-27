"""The model carried along the chain: encoder, alignment decoder, conformal scale sigma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..conformal.learned import ScaleNet, TrainReport, scale_of, train_scale
from ..conformal.signals import (
    SIGNALS,
    SignalStats,
    standardize_signals,
    window_signals,
)
from ..data.datasets import (
    HybridClassificationDataset,
    LatentSequenceDataset,
    hybrid_collate_fn,
)
from ..models.alignment import AlignmentDecoder
from ..models.config import (
    BATCH_SIZE,
    KL_WARMUP_EPOCHS,
    LATENT_DIM,
    STEP1_LEARNING_RATE,
    STEP2_LEARNING_RATE,
    STEP2_SEQ_LEN,
)
from ..models.encoder import SharedTemporalClassifier
from ..models.latents import (
    extract_aligned_latents,
    extract_latents_hybrid,
    extract_logvar,
)
from ..models.training import train_step1, train_step2
from ..utils.pipeline_prep import compute_class_counts
from ..utils.production_config import (
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

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class TrainBudget:
    """Epoch caps and early-stopping patience for the three trained stages.

    Attributes:
        epochs:       epoch cap for Step 1 and for Step 2.
        patience:     early-stopping patience for Step 1 and for Step 2.
        mlp_epochs:   epoch cap for the conformal scale sigma.
        mlp_patience: early-stopping patience for sigma.
        kl_warmup_epochs: length of Step 1's KL warm-up, run only when training from scratch.
    """

    epochs: int = 200
    patience: int = 20
    mlp_epochs: int = 300
    mlp_patience: int = 30
    kl_warmup_epochs: int = KL_WARMUP_EPOCHS


class ChainModel:
    """Encoder + alignment decoder + conformal scale, trained session by session.

    Built for the chain's first session; every later session is added when it arrives
    (``SharedTemporalClassifier.add_session``), so nothing about it depends on sessions that come
    after. Sigma and its signal standardization are replaced by each ``fit``.

    Args:
        first_session: the chain's first session.
        config:        its neuron config from ``build_hybrid_neuron_mapping``.
        n_classes:     number of regions.
        device:        device the networks live on.

    Attributes:
        step1:    the BC-VAE encoder (``SharedTemporalClassifier``).
        dec:      the alignment decoder.
        sessions: sessions whose input layers exist, in arrival order.
        sigma:    the deployed conformal scale, or None before the first ``fit`` (and in a
                  baseline model, which never trains one).
        stats:    the signal standardization sigma was trained with.
        history:  the last ``fit``'s training curves, ``{"step1", "step2", "sigma"}``, each
                  ``{"epochs", "best_epoch", "train_loss", "val_loss"}``: ``val_loss`` is the
                  held-out loss early stopping watched (Step 2: its region loss),
                  ``best_epoch`` the 1-based epoch restored (None when none was); ``sigma`` is
                  None when sigma was not trained.
    """

    def __init__(
        self,
        first_session: str,
        config: dict,
        n_classes: int,
        device: torch.device,
    ) -> None:
        self.step1 = SharedTemporalClassifier(
            n_shared=0, session_configs={first_session: config}, n_classes=n_classes
        ).to(device)
        self.dec = AlignmentDecoder(
            latent_dim=LATENT_DIM,
            n_classes=n_classes,
            seq_len=STEP2_SEQ_LEN,
            dropout=STEP2_DROPOUT,
        ).to(device)
        self.sessions = [first_session]
        self.sigma: ScaleNet | None = None
        self.stats: SignalStats | None = None
        self.history: dict[str, dict[str, Any] | None] = {}
        self.n_classes, self.device = n_classes, device

    def add_session(self, session: str, config: dict) -> None:
        """Create a newly arrived session's input layer and read-out."""
        if session not in self.sessions:
            self.step1.add_session(session, config)
            self.sessions.append(session)

    # ------------------------------------------------------------------ training

    def fit(
        self,
        s: str,
        d: dict,
        *,
        from_scratch: bool,
        budget: TrainBudget,
        alpha: float,
        seed: int,
        train_sigma: bool = True,
    ) -> dict[str, Any]:
        """Train (or fine-tune) the stages on session ``s``'s training block.

        Early stopping uses the session's calibration block (``val_idx``). The KL warm-up runs
        only when training from scratch. Sigma is fine-tuned from the previous session's sigma.
        The session's Step 1 latents are stored in ``d["step1_all_latents"]`` and the three
        stages' training curves in ``history``.

        Args:
            s:            the session's name.
            d:            its prepared data dict.
            from_scratch: True for the chain's first session (KL warm-up on, sigma from 1).
            budget:       epoch caps and patience.
            alpha:        miscoverage level sigma is trained for.
            seed:         seed for sigma's training.
            train_sigma:  False skips sigma (the standard split-conformal baseline); the
                          signal stats are still fitted.

        Returns:
            The training-block and calibration-block signal inputs sigma was fit on (for the
            coverage-sweep branches): ``{"tr", "va"}``, each ``{"u", "probs", "labels"}``.
        """
        batch = min(BATCH_SIZE, 256)
        # Train on this session alone, weighting classes by its own training block.
        one = {s: d}
        counts = compute_class_counts(one, [s])
        silent: Callable[[str], None] = lambda _m: None  # noqa: E731
        # A log interval longer than the run: no progress lines.
        quiet = budget.epochs + 1
        self.step1, hist1 = train_step1(
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
            log_fn=silent,
            n_classes=self.n_classes,
            num_epochs=budget.epochs,
            lr=STEP1_LEARNING_RATE,
            classifier_weight_decay=STEP1_WEIGHT_DECAY,
            region_weight=STEP1_REGION_WEIGHT,
            class_weight_power=STEP1_CLASS_WEIGHT_POWER,
            class_counts=counts,
            kl_weight=STEP1_KL_WEIGHT,
            kl_free_bits=STEP1_KL_FREE_BITS,
            kl_warmup_epochs=budget.kl_warmup_epochs if from_scratch else 0,
            early_stop_patience=budget.patience,
            log_every=quiet,
        )
        d["step1_all_latents"] = extract_latents_hybrid(
            self.step1, d["windows"], s, self.device
        )
        self.dec, hist2 = train_step2(
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
            log_fn=silent,
            n_classes=self.n_classes,
            num_epochs=budget.epochs,
            lr=STEP2_LEARNING_RATE,
            class_counts=counts,
            class_weight_power=STEP2_CLASS_WEIGHT_POWER,
            contrastive_weight=STEP2_CONTRASTIVE_WEIGHT,
            early_stop_patience=budget.patience,
            log_every=quiet,
        )
        fw = self.forward(s, d)
        inputs = {
            k: {
                "u": self.signals(s, d, fw, d[idx]),
                "probs": fw["probs"][d[idx]],
                "labels": d["regions"][d[idx]],
            }
            for k, idx in (("tr", "train_idx"), ("va", "val_idx"))
        }
        self.stats = SignalStats.fit(inputs["tr"]["u"])
        self.history = {
            "step1": _curve(hist1, "val_loss"),
            "step2": _curve(hist2, "val_region"),
            "sigma": None,
        }
        if train_sigma:
            self.sigma, report = self._fit_sigma(
                inputs, alpha, None if from_scratch else self.sigma, budget, seed
            )
            self.history["sigma"] = {
                "epochs": report.epochs,
                "best_epoch": None
                if report.best_epoch is None
                else report.best_epoch + 1,
                "train_loss": [float(v) for v in report.loss],
                "val_loss": [float(v) for v in report.earlystop_loss or []],
            }
        return inputs

    def train_sigma(
        self,
        inputs: dict[str, dict[str, np.ndarray]],
        alpha: float,
        init: ScaleNet | None,
        budget: TrainBudget,
        seed: int,
    ) -> ScaleNet:
        """Fit sigma on the training-block inputs, early-stopped on the calibration block.

        Args:
            inputs: ``{"tr", "va"}`` signal inputs from ``fit``.
            alpha:  miscoverage level to train for.
            init:   sigma to fine-tune from, or None to start at sigma = 1.
            budget: epoch cap and patience for sigma.
            seed:   training seed.

        Returns:
            The trained scale network.
        """
        return self._fit_sigma(inputs, alpha, init, budget, seed)[0]

    def _fit_sigma(
        self,
        inputs: dict[str, dict[str, np.ndarray]],
        alpha: float,
        init: ScaleNet | None,
        budget: TrainBudget,
        seed: int,
    ) -> tuple[ScaleNet, TrainReport]:
        """``train_sigma`` plus the training report (loss curves, restored epoch)."""
        assert self.stats is not None
        tr, va = inputs["tr"], inputs["va"]
        net, report = train_scale(
            standardize_signals(tr["u"], self.stats),
            tr["probs"],
            tr["labels"],
            alpha=alpha,
            hidden=MLP_HIDDEN,
            depth=MLP_DEPTH,
            epochs=budget.mlp_epochs,
            tau=MLP_TAU,
            kappa=MLP_KAPPA,
            size_weight=MLP_SIZE_WEIGHT,
            val_signals_z=standardize_signals(va["u"], self.stats),
            val_probs=va["probs"],
            val_labels=va["labels"],
            early_stop_patience=budget.mlp_patience,
            n_classes=self.n_classes,
            split=MLP_CAL_SPLIT,
            # One session: a single group, so the chronological split halves it in time.
            groups=np.zeros(len(tr["labels"]), int),
            val_groups=np.zeros(len(va["labels"]), int),
            init_net=init,
            seed=seed,
            device=str(self.device),
        )
        return net, report

    # ------------------------------------------------------------------ inference

    def forward(self, s: str, d: dict) -> dict[str, np.ndarray]:
        """Latents, log-variances and softmax probabilities for every window of a session.

        Args:
            s: the session's name.
            d: its prepared data dict.

        Returns:
            ``{"latents", "logvar", "probs"}``, one row per window of the session.
        """
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
        """The four uncertainty signals (``SIGNALS`` order) for the windows ``idx``.

        Args:
            s:   the session's name.
            d:   its prepared data dict.
            fw:  ``forward`` output for the session.
            idx: indices of the windows.

        Returns:
            (len(idx), 4) raw signals.
        """
        return window_signals(
            self.step1,
            self.dec,
            session=s,
            windows=d["windows"],
            latents=fw["latents"],
            logvar=fw["logvar"],
            probs=fw["probs"],
            idx=idx,
            seq_len=STEP2_SEQ_LEN,
            device=self.device,
        )

    def block(self, s: str, d: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Probabilities, raw signals and labels of the windows ``idx``, under the current model.

        Args:
            s:   the session's name.
            d:   its prepared data dict.
            idx: indices of the windows.

        Returns:
            ``{"probs", "u", "labels"}``, with ``u`` the raw signals.
        """
        fw = self.forward(s, d)
        return {
            "probs": fw["probs"][idx],
            "u": self.signals(s, d, fw, idx),
            "labels": d["regions"][idx],
        }

    def score(
        self, blk: dict[str, np.ndarray], sigma: ScaleNet | None = None
    ) -> dict[str, np.ndarray]:
        """Sigma and true-class scores ``(1 - p_y) / sigma`` for a block.

        ``sigma`` defaults to the deployed sigma; pass a sweep branch to score with it. With no
        deployed sigma (a baseline model) the scale is 1, i.e. standard split conformal.

        Args:
            blk:   a ``block`` output.
            sigma: the scale to score with, or None for the deployed one.

        Returns:
            ``{"probs", "sigma", "labels", "scores"}``.
        """
        net = sigma if sigma is not None else self.sigma
        if net is None:
            sig = np.ones(len(blk["labels"]))
        else:
            assert self.stats is not None
            sig = scale_of(
                net, standardize_signals(blk["u"], self.stats), device=str(self.device)
            )
        p_true = blk["probs"][np.arange(len(blk["labels"])), blk["labels"]]
        return {
            "probs": blk["probs"],
            "sigma": sig,
            "labels": blk["labels"],
            "scores": (1.0 - p_true) / sig,
        }

    # ------------------------------------------------------------------ checkpoints

    def state(self) -> dict[str, Any]:
        """Everything needed to rebuild the model (sessions, weights, sigma, stats).

        Returns:
            ``{"sessions", "step1", "dec", "sigma", "stats"}``: state dicts for the three
            networks (``sigma`` None if untrained) and the signal stats as
            ``(mean, std)`` lists.
        """
        return {
            "sessions": list(self.sessions),
            "step1": self.step1.state_dict(),
            "dec": self.dec.state_dict(),
            "sigma": None if self.sigma is None else self.sigma.state_dict(),
            "stats": None
            if self.stats is None
            else (self.stats.mean.tolist(), self.stats.std.tolist()),
        }

    def load(self, state: dict[str, Any], configs: dict) -> None:
        """Restore a ``state()``: re-add its sessions, then load the weights.

        Args:
            state:   a ``state()`` output.
            configs: neuron configs for every session in ``state["sessions"]``.
        """
        for s in state["sessions"]:
            self.add_session(s, configs[s])
        self.step1.load_state_dict(state["step1"])
        self.dec.load_state_dict(state["dec"])
        if state["sigma"] is not None:
            self.sigma = ScaleNet(len(SIGNALS), hidden=MLP_HIDDEN, depth=MLP_DEPTH).to(
                self.device
            )
            self.sigma.load_state_dict(state["sigma"])
        if state["stats"] is not None:
            self.stats = SignalStats(
                np.asarray(state["stats"][0]), np.asarray(state["stats"][1])
            )


def _curve(hist: dict, val_key: str) -> dict[str, Any]:
    """A training history (``train_step1`` / ``train_step2``) as a compact curve record.

    ``val_key`` names the held-out loss that early stopping watched, stored as ``val_loss``.
    """
    return {
        "epochs": len(hist.get("loss", [])),
        "best_epoch": hist.get("best_epoch"),
        "train_loss": [float(v) for v in hist.get("loss", [])],
        "val_loss": [None if v is None else float(v) for v in hist.get(val_key, [])],
    }
