"""Tests for src/conformal/strain.py."""

from __future__ import annotations

import numpy as np
import torch

from src.conformal.strain import compute_spectral_norms
from src.models.alignment import AlignmentDecoder

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
