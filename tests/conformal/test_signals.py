"""Tests for src/conformal/signals.py."""

from __future__ import annotations

import numpy as np
import torch

from src.conformal.signals import (
    SignalStats,
    compute_window_signals,
    standardize_signals,
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


def _session(n=6):
    """Return a minimal session_data entry with the fields compute_window_signals reads."""
    rng = np.random.default_rng(1)
    probs = rng.random((n, _NC))
    probs /= probs.sum(1, keepdims=True)
    return {
        "regions": rng.integers(0, _NC, n),
        "probs": probs,
        "logvar": rng.standard_normal((n, _LD)).astype(np.float32),
        "step1_all_latents": rng.standard_normal((n, _LD)).astype(np.float32),
        "windows": rng.standard_normal((n, 3, 4)).astype(np.float32),
    }


class TestComputeWindowSignals:
    def test_entropy_column_matches_manual(self):
        """Signal column 0 is the Shannon entropy of each window's posterior, computed by hand."""
        sd = {"S1": _session(6)}
        compute_window_signals(
            _encoder(), _decoder(), sd, ["S1"], seq_len=_SL, device=_DEVICE
        )
        p = sd["S1"]["probs"].astype(np.float64)
        expected = -(p * np.log(np.clip(p, 1e-10, 1.0))).sum(1)
        assert np.allclose(sd["S1"]["signals"][:, 0], expected)

    def test_variance_column_matches_manual(self):
        """Signal column 1 is the mean exponentiated posterior log-variance per window."""
        sd = {"S1": _session(6)}
        compute_window_signals(
            _encoder(), _decoder(), sd, ["S1"], seq_len=_SL, device=_DEVICE
        )
        expected = np.exp(sd["S1"]["logvar"].astype(np.float64)).mean(1)
        assert np.allclose(sd["S1"]["signals"][:, 1], expected)


class TestSignalStats:
    def test_frozen_stats_applied_to_new_data(self):
        """Standardization must reuse the fitted statistics, never recompute on the
        block being scored. Recomputing would leak test-block information."""
        stats = SignalStats.fit(np.array([[0.0], [2.0]]))  # mean 1, std 1
        z = standardize_signals(np.array([[1.0], [3.0]]), stats)
        assert np.allclose(z.ravel(), [0.0, 2.0])
