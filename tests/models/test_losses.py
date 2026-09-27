"""Tests for src/models/losses.py.

Accepted gaps: none.
"""

import math

import pytest
import torch

from src.models.losses import compute_contrastive_loss, poisson_nll


def test_poisson_nll_matches_its_closed_form():
    """Poisson NLL matches its closed form at rate=e, target=1."""
    rate = math.e
    expected = rate - math.log(rate + 1e-8)
    got = poisson_nll(torch.full((1, 1, 1), rate), torch.ones(1, 1, 1)).item()
    assert abs(got - expected) < 1e-4


def test_nt_xent_closed_form_for_identical_projections():
    """NT-Xent over four identical unit vectors equals log(3) at temperature 1."""
    # 4 identical unit vectors, same region, same session.
    # Positive pairs: 3 non-self per row (normalized weight 1/3 each).
    # weighted_pos = exp(1/T), all_sum = 3*exp(1/T) -> loss = log(3) at T=1.
    proj = torch.zeros(4, 4)
    proj[:, 0] = 1.0
    loss = compute_contrastive_loss(proj, [0] * 4, [0] * 4, temperature=1.0)
    assert abs(loss.item() - math.log(3)) < 1e-4


def test_poisson_nll_is_minimized_at_the_observed_count():
    """d/d rate (rate - k log rate) = 0 at rate = k: the loss is lowest when rate matches."""
    target = torch.full((1, 1, 1), 3.0)
    losses = [
        poisson_nll(torch.full((1, 1, 1), r), target).item() for r in (2.0, 3.0, 4.0)
    ]
    assert losses[1] < losses[0] and losses[1] < losses[2]


def test_poisson_nll_survives_a_zero_rate():
    """A zero predicted rate with zero spikes is finite thanks to the eps floor."""
    assert math.isfinite(poisson_nll(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)).item())


def test_nt_xent_weights_cross_session_positives_double():
    """Rows 0 and 1 are identical (session 0), row 2 orthogonal (session 1), all one region,
    temperature 1. After the per-row max shift, row 0 sees exp-similarities (1, e^-1) to rows
    1 and 2; its positives are weighted 1 (same session) and 2 (cross), normalized to 1/3 and
    2/3. Row 2 sees e^-1 twice, both cross-session, so its ratio is 1/2."""
    proj = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    e = math.exp(-1.0)
    row01 = -math.log((1 / 3 + 2 / 3 * e) / (1 + e))
    expected = (2 * row01 + math.log(2.0)) / 3
    loss = compute_contrastive_loss(proj, [0, 0, 0], [0, 0, 1], temperature=1.0)
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_nt_xent_ignores_other_regions_as_positives():
    """A different-region neighbor is only a negative: with one same-region pair the loss is
    the softmax share of that pair."""
    proj = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # rows 0,1 region 0 (positive pair, sim 1 -> shifted 0), row 2 region 1 (sim 0 -> -1).
    e = math.exp(-1.0)
    loss = compute_contrastive_loss(proj, [0, 0, 1], [0, 1, 0], temperature=1.0)
    row01 = -math.log(1 / (1 + e))
    row2 = -math.log(1e-8)  # no positive at all: only the epsilon remains
    assert loss.item() == pytest.approx((2 * row01 + row2) / 3, rel=1e-4)


def test_lower_temperature_sharpens_the_same_similarities():
    """Dividing similarities by a smaller temperature widens the gap to negatives, so a
    well-separated batch has lower loss at T = 0.1 than at T = 1."""
    proj = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels, sessions = [0, 0, 1, 1], [0, 1, 0, 1]
    hot = compute_contrastive_loss(proj, labels, sessions, temperature=1.0)
    cold = compute_contrastive_loss(proj, labels, sessions, temperature=0.1)
    assert cold.item() < hot.item()
