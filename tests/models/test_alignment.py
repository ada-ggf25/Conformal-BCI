"""Tests for src/models/alignment.py."""

import torch

from src.models.alignment import AlignmentDecoder

_LD, _NC, _SL, _DM, _HD, _BATCH = 4, 3, 3, 8, 8, 2


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


def test_proj_is_l2_normalized():
    """The contrastive projection head returns unit-norm vectors."""
    _, proj, _ = _decoder()(torch.randn(_BATCH, _SL, _LD))
    torch.testing.assert_close(
        proj.norm(dim=-1), torch.ones(_BATCH), atol=1e-5, rtol=0.0
    )


def test_align_matches_z_aligned_from_forward():
    """align() returns exactly the aligned latent that forward() produces."""
    model = _decoder().eval()
    x = torch.randn(_BATCH, _SL, _LD)
    with torch.no_grad():
        z_aligned, _, _ = model(x)
        torch.testing.assert_close(z_aligned, model.align(x))
