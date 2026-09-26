"""Tests for src/conformal/signals.py."""

from __future__ import annotations

import numpy as np
import torch

from src.conformal.signals import (
    SignalStats,
    standardize_signals,
    window_signals,
)
from src.models.alignment import AlignmentDecoder
from src.models.encoder import SharedTemporalClassifier

_LD, _SL, _DM, _HD, _NC = 4, 3, 8, 8, 2
_DEVICE = torch.device("cpu")


def _encoder() -> SharedTemporalClassifier:
    """Build a tiny single-session BC-VAE encoder."""
    configs = {
        "S1": {
            "n_total": 4,
            "shared_indices": [0, 1],
            "specific_indices": [2, 3],
            "n_specific": 2,
        }
    }
    return SharedTemporalClassifier(
        2, configs, _NC, shared_dim=_DM, latent_dim=_LD, factor_dim=4, hidden_dim=_HD
    )


def _decoder() -> AlignmentDecoder:
    """Build a tiny Transformer alignment decoder."""
    return AlignmentDecoder(
        latent_dim=_LD,
        n_classes=_NC,
        seq_len=_SL,
        n_heads=2,
        n_layers=1,
        d_model=_DM,
        hidden_dim=_HD,
    )


def _signals(n=6, idx=(1, 3, 4)):
    """Signals for a subset of a random session's windows, plus the arrays they came from."""
    rng = np.random.default_rng(1)
    probs = rng.random((n, _NC))
    probs /= probs.sum(1, keepdims=True)
    arrays = {
        "probs": probs,
        "logvar": rng.standard_normal((n, _LD)).astype(np.float32),
        "latents": rng.standard_normal((n, _LD)).astype(np.float32),
        "windows": rng.standard_normal((n, 3, 4)).astype(np.float32),
    }
    idx = np.array(idx)
    u = window_signals(
        _encoder(),
        _decoder(),
        session="S1",
        idx=idx,
        seq_len=_SL,
        device=_DEVICE,
        **arrays,
    )
    return u, arrays, idx


class TestWindowSignals:
    def test_entropy_column_matches_manual(self):
        """Signal column 0 is the Shannon entropy of each selected window's posterior."""
        u, a, idx = _signals()
        p = a["probs"][idx]
        assert np.allclose(u[:, 0], -(p * np.log(np.clip(p, 1e-10, 1.0))).sum(1))

    def test_variance_column_matches_manual(self):
        """Signal column 1 is the mean exponentiated posterior log-variance per selected window."""
        u, a, idx = _signals()
        assert np.allclose(u[:, 1], np.exp(a["logvar"][idx].astype(np.float64)).mean(1))


class TestSignalStats:
    def test_frozen_stats_applied_to_new_data(self):
        """Standardization must reuse the fitted statistics, never recompute on the
        block being scored. Recomputing would leak test-block information."""
        stats = SignalStats.fit(np.array([[0.0], [2.0]]))  # mean 1, std 1
        z = standardize_signals(np.array([[1.0], [3.0]]), stats)
        assert np.allclose(z.ravel(), [0.0, 2.0])
