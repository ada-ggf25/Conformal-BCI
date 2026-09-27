"""Tests for src/conformal/signals.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.conformal.signals import (
    SignalStats,
    standardize_signals,
    window_signals,
)
from src.conformal.strain import compute_encoder_spectral_norms, compute_spectral_norms
from src.models.alignment import AlignmentDecoder
from src.models.encoder import SharedTemporalClassifier
from src.models.latents import build_latent_sequences

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
    torch.manual_seed(0)
    enc, dec = _encoder(), _decoder()
    arrays["models"] = (enc, dec)
    u = window_signals(
        enc,
        dec,
        session="S1",
        idx=idx,
        seq_len=_SL,
        device=_DEVICE,
        **{k: v for k, v in arrays.items() if k != "models"},
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

    def test_alignment_jacobian_reads_each_windows_true_context(self):
        """Column 2 differentiates the aligned latent over the window's real preceding
        latents (from the whole session), not over a sequence rebuilt from idx alone."""
        u, a, idx = _signals()
        _, dec = a["models"]
        full = compute_spectral_norms(
            dec, build_latent_sequences(a["latents"], _SL)[idx], _DEVICE
        )
        only_idx = compute_spectral_norms(
            dec, build_latent_sequences(a["latents"][idx], _SL), _DEVICE
        )
        np.testing.assert_allclose(u[:, 2], full, rtol=1e-6)
        assert not np.allclose(u[:, 2], only_idx)

    def test_encoder_jacobian_is_taken_on_the_selected_windows(self):
        """Column 3 is the encoder Jacobian norm of exactly the windows idx."""
        u, a, idx = _signals()
        enc, _ = a["models"]
        want = compute_encoder_spectral_norms(enc, a["windows"][idx], "S1", _DEVICE)
        np.testing.assert_allclose(u[:, 3], want, rtol=1e-6)


class TestSignalStats:
    def test_frozen_stats_applied_to_new_data(self):
        """Standardization must reuse the fitted statistics, never recompute on the
        block being scored. Recomputing would leak test-block information."""
        stats = SignalStats.fit(np.array([[0.0], [2.0]]))  # mean 1, std 1
        z = standardize_signals(np.array([[1.0], [3.0]]), stats)
        assert np.allclose(z.ravel(), [0.0, 2.0])

    def test_fit_statistics_are_per_column(self):
        """Each signal is centered and scaled by its own column's mean and std."""
        stats = SignalStats.fit(np.array([[0.0, 10.0], [2.0, 30.0], [4.0, 50.0]]))
        np.testing.assert_allclose(stats.mean, [2.0, 30.0])
        np.testing.assert_allclose(stats.std, np.sqrt([8 / 3, 800 / 3]), rtol=1e-6)

    def test_a_constant_signal_does_not_divide_by_zero(self):
        """A signal with no spread standardizes to 0, not NaN or inf."""
        stats = SignalStats.fit(np.full((4, 1), 7.0))
        assert np.all(standardize_signals(np.full((2, 1), 7.0), stats) == 0.0)

    def test_fit_needs_a_matrix(self):
        """A 1-D vector is ambiguous (one signal or one window?) and is refused."""
        with pytest.raises(ValueError, match="must be"):
            SignalStats.fit(np.zeros(4))

    @pytest.mark.parametrize("shape", [(3,), (3, 2)])
    def test_standardize_refuses_a_different_signal_count(self, shape):
        """Stats of one signal cannot be applied to a vector or to two signals."""
        stats = SignalStats.fit(np.zeros((4, 1)))
        with pytest.raises(ValueError, match="incompatible"):
            standardize_signals(np.zeros(shape), stats)
