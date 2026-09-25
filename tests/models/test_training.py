"""Tests for src/models/training.py."""

from __future__ import annotations

import pytest
import torch

from src.models.alignment import AlignmentDecoder
from src.models.config import RECON_WEIGHT
from src.models.encoder import SharedTemporalClassifier
from src.models.training import train_step1, train_step2

_LD, _DM, _HD, _NC, _BATCH, _T, _SL = 4, 8, 8, 3, 2, 3, 3
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


def _step1_batch():
    """Return one Step-1 batch shaped as hybrid_collate_fn emits it."""
    # Mirrors hybrid_collate_fn: windows_raw is the Poisson reconstruction target.
    return {
        "S1": {
            "windows": torch.zeros(_BATCH, _T, 4),
            "windows_raw": torch.zeros(_BATCH, _T, 4),
            "regions": torch.zeros(_BATCH, dtype=torch.long),
        }
    }


def _step2_batch():
    """Return one Step-2 batch of (sequences, regions, session ids)."""
    return (
        torch.zeros(_BATCH, _SL, _LD),
        torch.zeros(_BATCH, dtype=torch.long),
        torch.zeros(_BATCH, dtype=torch.long),
    )


def test_total_loss_is_the_documented_weighted_sum():
    """Loss = RECON_WEIGHT*recon + region_weight*region + kl_w*kl.

    Recomposing it from the recorded components pins the weighting itself. A
    region_weight silently ignored (the failure that lets the decoder learn the
    majority region and nothing else) shows up here as an arithmetic mismatch.
    """
    torch.manual_seed(0)
    _, hist = train_step1(
        _encoder(),
        [_step1_batch()],  # one batch, so the epoch means are exact
        _DEVICE,
        lambda s: None,
        _NC,
        num_epochs=1,
        lr=1e-3,
        region_weight=2.5,
    )
    expected = (
        RECON_WEIGHT * hist["recon"][0]
        + 2.5 * hist["region"][0]
        + hist["kl_weight"][0] * hist["kl"][0]
    )
    assert hist["loss"][0] == pytest.approx(expected, rel=1e-5)


def test_train_step2_restores_best_val_weights():
    """train_step2 returns the same decoder carrying best-val weights, and logs the restore."""
    # With a val_loader the best-val weights must be restored before return,
    # rather than the final / early-stopped epoch's.
    logs: list[str] = []
    decoder = _decoder()
    out_dec, _ = train_step2(
        decoder,
        [_step2_batch()],
        _DEVICE,
        logs.append,
        _NC,
        num_epochs=10,
        lr=3e-4,
        val_loader=[_step2_batch()],
        early_stop_patience=1,
        early_stop_min_delta=1e10,
    )
    assert out_dec is decoder
    assert any("restored best-val weights" in m for m in logs)


def test_rare_class_gets_proportionally_more_weight():
    """Full inverse-frequency weighting is what stops the decoder
    from simply learning the majority region."""
    from src.models.training import _inverse_frequency_weights

    w = _inverse_frequency_weights({0: 900, 1: 100}, 2, power=1.0, device=_DEVICE)
    assert w[1] / w[0] == pytest.approx(9.0)
    assert w.sum().item() == pytest.approx(2.0)  # renormalized to n_classes
