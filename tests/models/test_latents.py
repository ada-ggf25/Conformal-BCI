"""Tests for src/models/latents.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.encoder import SharedTemporalClassifier
from src.models.latents import (
    build_latent_sequences,
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
}


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
