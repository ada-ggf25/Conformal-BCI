"""The four per-window uncertainty signals the conformal scale scores on.

Listed in ``SIGNALS`` column order, which is the order they are persisted in:

    entropy         - a function of the softmax decoder predictions
    variance        - the BC-VAE posterior variance, exp(logvar).mean
    align_jacobian  - ||d z_aligned / d z||, the Step-2 alignment map's spectral norm
    enc_jacobian    - ||d mu / d x||, the Step-1 encoder map's spectral norm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..models.latents import build_latent_sequences
from .strain import compute_encoder_spectral_norms, compute_spectral_norms

if TYPE_CHECKING:
    import torch

    from ..models.alignment import AlignmentDecoder
    from ..models.encoder import SharedTemporalClassifier

SIGNALS = ("entropy", "variance", "align_jacobian", "enc_jacobian")
"""Canonical order of the four per-window uncertainty signals the learned scale reads."""


def window_signals(
    step1_model: SharedTemporalClassifier,
    alignment_decoder: AlignmentDecoder,
    *,
    session: str,
    windows: np.ndarray,
    latents: np.ndarray,
    logvar: np.ndarray,
    probs: np.ndarray,
    idx: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """The four signals (``SIGNALS`` order) for the windows ``idx`` of one session.

    ``windows``, ``latents``, ``logvar`` and ``probs`` are the session's full per-window arrays:
    the alignment Jacobian reads each window's preceding latents as context.

    Returns:
        (len(idx), 4) float64 array.
    """
    p = probs[idx].astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-10, 1.0))).sum(axis=1)
    var = np.exp(logvar[idx].astype(np.float64)).mean(axis=1)
    seqs = build_latent_sequences(latents, seq_len)[idx]
    aj = compute_spectral_norms(alignment_decoder, seqs, device)
    ej = compute_encoder_spectral_norms(step1_model, windows[idx], session, device)
    return np.column_stack([ent, var, aj, ej]).astype(np.float64)


@dataclass(frozen=True)
class SignalStats:
    """Per-signal centering/scaling, fitted once on a reference block and then frozen."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, signals: np.ndarray) -> SignalStats:
        """Fit per-signal mean and std on a reference block, then freeze them.

        Fit this once (on the block the scale is trained on) and pass the result to
        every later ``standardize_signals`` call. A tiny epsilon is added to each std
        so a constant signal cannot divide by zero.

        Args:
            signals: (n, n_signals) raw signal matrix in ``SIGNALS`` column order.

        Raises:
            ValueError: if ``signals`` is not 2-D.
        """
        s = np.asarray(signals, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"signals must be (n, n_signals), got shape {s.shape}")
        return cls(mean=s.mean(axis=0), std=s.std(axis=0) + 1e-8)


def standardize_signals(signals: np.ndarray, stats: SignalStats) -> np.ndarray:
    """Apply a frozen SignalStats map to a signal matrix -> (n, n_signals) float64."""
    s = np.asarray(signals, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != stats.mean.shape[0]:
        raise ValueError(
            f"signals {s.shape} incompatible with stats of {stats.mean.shape[0]} signals"
        )
    return (s - stats.mean) / stats.std
