"""Tests for src/models/losses.py."""

import math

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
