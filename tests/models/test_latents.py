"""Tests for src/models/latents.py.

Accepted gaps: none.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.alignment import AlignmentDecoder
from src.models.encoder import SharedTemporalClassifier
from src.models.latents import (
    build_latent_sequences,
    extract_aligned_latents,
    extract_latents_hybrid,
    extract_logvar,
)

_LD, _DM, _HD, _NC = 4, 8, 8, 3
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


# These helpers loop over batches and reassemble the outputs.
_WINDOWS = np.random.default_rng(0).standard_normal((9, 3, 4)).astype(np.float32)
_BATCHED = {
    "extract_latents_hybrid": lambda enc, bs: extract_latents_hybrid(
        enc, _WINDOWS, "S1", _DEVICE, batch_size=bs
    ),
    "extract_logvar": lambda enc, bs: extract_logvar(
        enc, _WINDOWS, "S1", _DEVICE, batch_size=bs
    ),
    "extract_aligned_latents": lambda _enc, bs: extract_aligned_latents(
        _decoder(), _WINDOWS[:, 0, :], _DEVICE, seq_len=3, batch_size=bs
    ),
}


def _decoder() -> AlignmentDecoder:
    """Build a tiny, deterministically initialized alignment decoder over 4-dim latents."""
    torch.manual_seed(1)
    return AlignmentDecoder(
        latent_dim=4, n_classes=_NC, seq_len=3, n_heads=2, n_layers=1, d_model=_DM,
        hidden_dim=_HD,
    )  # fmt: skip


@pytest.mark.parametrize("helper", list(_BATCHED))
def test_batching_does_not_change_result(helper):
    """Splitting the input into ragged batches gives the same result as a single batch."""
    torch.manual_seed(0)
    enc = _encoder().eval()
    call = _BATCHED[helper]
    np.testing.assert_allclose(call(enc, 10_000), call(enc, 4), rtol=1e-4, atol=1e-5)


def test_first_window_is_front_padded_with_its_own_latent():
    """The first window repeats its own latent in the sequence slots that have no history."""
    latents = np.arange(12, dtype=np.float32).reshape(3, 4)
    seqs = build_latent_sequences(latents, seq_len=3)
    for i in range(2):
        np.testing.assert_array_equal(seqs[0, i], latents[0])


def test_extract_logvar_returns_the_logvar_not_the_latent_mean():
    """The conformal "variance" signal reads this. Returning mu instead would be
    silent (the shape is identical) so pin it against the encoder's own output.
    """
    enc = _encoder().eval()
    w = np.random.default_rng(0).standard_normal((5, 3, 4)).astype(np.float32)
    got = extract_logvar(enc, w, "S1", _DEVICE)
    with torch.no_grad():
        mu, logvar = enc.get_latent_and_logvar(torch.from_numpy(w), "S1")
    assert np.allclose(got, logvar.numpy(), atol=1e-5)
    assert not np.allclose(got, mu.numpy())


def test_every_sequence_ends_at_its_own_window():
    """Sequence i holds latents i-2, i-1, i (clamped to the first at the start)."""
    latents = np.arange(5, dtype=np.float32)[:, None]
    seqs = build_latent_sequences(latents, seq_len=3)[:, :, 0]
    assert seqs.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]


def test_aligned_latents_are_the_align_map_on_those_sequences():
    """extract_aligned_latents is align() applied to build_latent_sequences, in eval mode."""
    dec = _decoder()
    lat = np.random.default_rng(3).standard_normal((6, 4)).astype(np.float32)
    got = extract_aligned_latents(dec, lat, _DEVICE, seq_len=3)
    with torch.no_grad():
        want = dec.eval().align(torch.from_numpy(build_latent_sequences(lat, 3)))
    np.testing.assert_allclose(got, want.numpy(), atol=1e-6)


def test_latents_are_the_posterior_mean():
    """Step 1 latents are mu, deterministic across calls (never a sample)."""
    enc = _encoder()
    a = extract_latents_hybrid(enc, _WINDOWS, "S1", _DEVICE)
    b = extract_latents_hybrid(enc, _WINDOWS, "S1", _DEVICE)
    np.testing.assert_array_equal(a, b)
    with torch.no_grad():
        mu = enc.get_latent(torch.from_numpy(_WINDOWS), "S1")
    np.testing.assert_allclose(a, mu.numpy(), atol=1e-6)
