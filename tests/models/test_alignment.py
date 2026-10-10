"""Tests for src/models/alignment.py.

Accepted gaps: output shapes (a wrong width stops the run loudly).
"""

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


def _swap_context(x: torch.Tensor) -> torch.Tensor:
    """The same sequence with its first two (context) latents swapped; the last is kept."""
    return x[:, [1, 0, 2], :]


def test_positional_embedding_makes_the_map_order_sensitive():
    """Self-attention is permutation-equivariant, so without a positional embedding swapping
    two context latents cannot change the last token's output; with it, it must."""
    torch.manual_seed(0)
    model = _decoder().eval()
    with torch.no_grad():
        model.pos_embed.normal_(0.0, 1.0)
    x = torch.randn(_BATCH, _SL, _LD)
    with torch.no_grad():
        assert not torch.allclose(
            model.align(x), model.align(_swap_context(x)), atol=1e-4
        )
        model.pos_embed.zero_()
        torch.testing.assert_close(model.align(x), model.align(_swap_context(x)))


def test_logits_are_the_decoder_head_on_the_aligned_latent():
    """Region logits are read from z_aligned, the same latent the conformal signals use."""
    model = _decoder().eval()
    x = torch.randn(_BATCH, _SL, _LD)
    with torch.no_grad():
        z, _, logits = model(x)
        torch.testing.assert_close(logits, model.decoder(z))
