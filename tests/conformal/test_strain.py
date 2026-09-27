"""Tests for src/conformal/strain.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import torch

from src.conformal.strain import compute_encoder_spectral_norms, compute_spectral_norms
from src.models.alignment import AlignmentDecoder
from src.models.encoder import SharedTemporalClassifier

_LD, _SL, _DM, _HD, _NC, _N = 4, 3, 8, 8, 2, 6
_DEVICE = torch.device("cpu")


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


def test_value_is_the_largest_singular_value_of_the_true_jacobian():
    """A function returning a constant, passes any shape or invariance check."""
    model = _decoder().eval()
    seqs = torch.from_numpy(
        np.random.default_rng(0).standard_normal((_N, _SL, _LD)).astype(np.float32)
    )
    got = compute_spectral_norms(model, seqs, _DEVICE)
    for i in range(len(seqs)):
        ctx = seqs[i : i + 1, :-1, :]

        def _align_of_last(z_last, ctx=ctx):
            """Align a sequence whose final latent is the variable being differentiated."""
            return model.align(torch.cat([ctx, z_last.view(1, 1, _LD)], dim=1))[0]

        j = torch.autograd.functional.jacobian(_align_of_last, seqs[i, -1, :].clone())
        expected = float(torch.linalg.svdvals(j.reshape(_LD, _LD))[0])
        assert abs(float(got[i]) - expected) < 1e-4


def _encoder() -> SharedTemporalClassifier:
    """Build a tiny single-session BC-VAE encoder."""
    configs = {
        "S1": {
            "n_total": 3,
            "shared_indices": [0],
            "specific_indices": [1, 2],
            "n_specific": 2,
        }
    }
    return SharedTemporalClassifier(
        1, configs, _NC, shared_dim=_DM, latent_dim=_LD, factor_dim=4, hidden_dim=_HD
    )


def test_encoder_value_is_the_largest_singular_value_of_d_mu_d_x():
    """The encoder signal is sigma_max of the (latent_dim x T*C) Jacobian of mu."""
    torch.manual_seed(0)
    enc = _encoder().eval()
    w = torch.randn(4, 2, 3)
    got = compute_encoder_spectral_norms(enc, w, "S1", _DEVICE)
    for i in range(len(w)):
        j = torch.autograd.functional.jacobian(
            lambda x: enc.get_latent(x.view(1, 2, 3), "S1")[0], w[i].clone()
        )
        expected = float(torch.linalg.svdvals(j.reshape(_LD, 6))[0])
        assert abs(float(got[i]) - expected) < 1e-4


def test_batching_and_input_type_do_not_change_the_norms():
    """Ragged batches and numpy input give the same per-row values as one tensor batch."""
    torch.manual_seed(0)
    dec, enc = _decoder().eval(), _encoder().eval()
    seqs = np.random.default_rng(1).standard_normal((5, _SL, _LD)).astype(np.float32)
    np.testing.assert_allclose(
        compute_spectral_norms(dec, seqs, _DEVICE, batch_size=2),
        compute_spectral_norms(dec, torch.from_numpy(seqs), _DEVICE, batch_size=64),
        rtol=1e-5,
    )
    w = np.random.default_rng(2).standard_normal((5, 2, 3)).astype(np.float32)
    np.testing.assert_allclose(
        compute_encoder_spectral_norms(enc, w, "S1", _DEVICE, batch_size=2),
        compute_encoder_spectral_norms(enc, torch.from_numpy(w), "S1", _DEVICE),
        rtol=1e-5,
    )


def test_only_the_last_latents_block_of_the_jacobian_is_used():
    """The norm is the last-latent block of the full-sequence Jacobian: the context latents
    are inputs, not differentiated variables."""
    torch.manual_seed(0)
    dec = _decoder().eval()
    seqs = torch.randn(1, _SL, _LD)
    base = compute_spectral_norms(dec, seqs, _DEVICE)
    j = torch.autograd.functional.jacobian(lambda s: dec.align(s)[0], seqs)
    last = j[:, 0, -1, :]
    assert abs(float(base[0]) - float(torch.linalg.svdvals(last)[0])) < 1e-4
