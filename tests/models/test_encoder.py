"""Tests for src/models/encoder.py.

Accepted gaps: output shapes (a wrong width stops the run loudly).
"""

import math

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


def test_adding_a_session_later_leaves_earlier_weights_untouched():
    """In a session chain a session's layers are created when it arrives: a model grown by
    add_session must keep every earlier weight exactly as a model built without that session."""
    cfg = _configs()
    torch.manual_seed(0)
    alone = _encoder({"S1": cfg["S1"]})
    torch.manual_seed(0)
    grown = _encoder({"S1": cfg["S1"]})
    grown.add_session("S2", cfg["S2"])
    before, after = alone.state_dict(), grown.state_dict()
    assert set(before) < set(after)
    for k, v in before.items():
        assert torch.equal(v, after[k]), k
    x = torch.randn(_BATCH, _T, 5)
    grown.eval()
    assert torch.isfinite(grown.get_latent(x, "S2")).all()


def test_inference_classifies_from_the_posterior_mean():
    """In eval mode the region logits come from mu, so repeated calls agree even though the
    sampled latent z differs; training mode classifies from the noisy z."""
    torch.manual_seed(0)
    model = _encoder()
    x = torch.randn(_BATCH, _T, 4)
    model.eval()
    a = model(x, "S1", return_all=True)
    b = model(x, "S1", return_all=True)
    torch.testing.assert_close(a["region_logits"], b["region_logits"])
    torch.testing.assert_close(a["region_logits"], model.classifier(a["mu"]))
    assert not torch.equal(a["z"], b["z"])


def test_reparameterize_is_mu_plus_scaled_noise():
    """z = mu + eps * exp(logvar / 2): with the same noise draw, it is exactly that."""
    model = _encoder()
    mu, logvar = torch.tensor([[1.0, -2.0]]), torch.tensor([[0.0, 2.0 * math.log(3.0)]])
    torch.manual_seed(5)
    z = model.reparameterize(mu, logvar)
    torch.manual_seed(5)
    eps = torch.randn(1, 2)
    torch.testing.assert_close(z, mu + eps * torch.tensor([[1.0, 3.0]]))


def test_forward_returns_the_sampled_latent_by_default():
    """Without return_all the forward pass is the sampled z, whose mean over draws is mu."""
    model = _encoder().eval()
    x = torch.randn(1, _T, 4)
    torch.manual_seed(0)
    draws = model(x.expand(4000, -1, -1), "S1")
    mu = model.get_latent(x, "S1")
    torch.testing.assert_close(draws.mean(0), mu[0], atol=0.1, rtol=0)


def test_poisson_rates_are_strictly_positive():
    """The read-out ends in Softplus, so no rate can reach 0 and blow up the NLL log."""
    out = _encoder()(torch.randn(3, _T, 4) * 50, "S1", return_all=True)
    assert (out["rates"] > 0).all()


def test_session_specific_neurons_reach_the_latent():
    """Changing only a session-specific column changes mu: that projection is wired in."""
    model = _encoder().eval()
    x = torch.randn(1, _T, 4)
    y = x.clone()
    y[..., 3] += 5.0  # column 3 is S1-specific
    assert not torch.allclose(model.get_latent(x, "S1"), model.get_latent(y, "S1"))


def test_get_latent_is_the_mu_of_get_latent_and_logvar():
    """Both inference entry points read the same posterior mean."""
    model = _encoder().eval()
    x = torch.randn(_BATCH, _T, 4)
    mu, _ = model.get_latent_and_logvar(x, "S1")
    torch.testing.assert_close(model.get_latent(x, "S1"), mu)
