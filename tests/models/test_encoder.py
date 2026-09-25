"""Tests for src/models/encoder.py."""

import torch

from src.models.encoder import SharedTemporalClassifier

_LD, _DM, _HD, _NC, _BATCH, _T = 4, 8, 8, 3, 2, 3


def _configs():
    """Return two session configs of differing width that share two neurons."""
    return {
        "S1": {
            "n_total": 4,
            "shared_indices": [0, 1],
            "specific_indices": [2, 3],
            "n_specific": 2,
        },
        "S2": {
            "n_total": 5,
            "shared_indices": [0, 1],
            "specific_indices": [2, 3, 4],
            "n_specific": 3,
        },
    }


def _encoder(configs=None):
    """Build a tiny pooled BC-VAE encoder over the given session configs."""
    return SharedTemporalClassifier(
        2,
        configs or _configs(),
        _NC,
        shared_dim=_DM,
        latent_dim=_LD,
        factor_dim=4,
        hidden_dim=_HD,
    )


def test_logvar_is_not_forced_positive():
    """The log-variance head passes a fixed negative value through unclamped."""
    model = _encoder()
    with torch.no_grad():
        model.encoder.fc_logvar.weight.zero_()
        model.encoder.fc_logvar.bias.fill_(-3.0)
    _, logvar = model.get_latent_and_logvar(torch.zeros(_BATCH, _T, 4), "S1")
    torch.testing.assert_close(logvar, torch.full_like(logvar, -3.0))


def test_each_session_routes_through_its_own_input_projection():
    """The pooled model's core mechanism: sessions of different widths, including
    one with no session-specific neurons at all, must all decode to n_classes."""
    out2 = _encoder()(torch.zeros(_BATCH, _T, 5), "S2", return_all=True)
    assert out2["region_logits"].shape == (_BATCH, _NC)

    no_spec = _encoder(
        {
            "Sno": {
                "n_total": 2,
                "shared_indices": [0, 1],
                "specific_indices": [],
                "n_specific": 0,
            }
        }
    )
    out = no_spec(torch.zeros(_BATCH, _T, 2), "Sno", return_all=True)
    assert out["region_logits"].shape == (_BATCH, _NC)
