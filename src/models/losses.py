"""Loss functions for Step 1 (BC-VAE) and Step 2 (Transformer alignment) training."""

import torch

from .config import TEMPERATURE


def poisson_nll(
    rates: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Poisson negative log-likelihood.

    NLL(rate, target) = rate - target * log(rate)

    Args:
        rates:   predicted Poisson rates, shape (batch, T, n_shared); must be > 0
        targets: observed spike counts, same shape
        eps:     floor added to rates to avoid log(0)

    Returns:
        scalar mean NLL
    """
    rates = rates + eps
    return (rates - targets * torch.log(rates)).mean()


def compute_contrastive_loss(
    proj: torch.Tensor,
    region_labels: list[int],
    session_ids: list[int],
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """NT-Xent contrastive loss for Step 2 alignment.

    Positive pairs are windows from the same region but different sessions.
    Cross-session same-region pairs receive double weight (1 + 1 = 2) relative to
    same-session same-region pairs (weight = 1). Self-pairs are excluded.

    Args:
        proj:          L2-normalized projections, shape (batch, proj_dim)
        region_labels: region index for each sample in the batch
        session_ids:   session index for each sample in the batch
        temperature:   softmax temperature

    Returns:
        scalar contrastive loss
    """
    batch_size = proj.shape[0]
    device = proj.device

    sim_matrix = torch.mm(proj, proj.t()) / temperature
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

    region_labels_t = torch.tensor(region_labels, device=device)
    same_region = region_labels_t.unsqueeze(0) == region_labels_t.unsqueeze(1)

    session_ids_t = torch.tensor(session_ids, device=device)
    cross_session = session_ids_t.unsqueeze(0) != session_ids_t.unsqueeze(1)

    positive_weights = same_region.float() * (1.0 + cross_session.float())

    mask_self = torch.eye(batch_size, device=device, dtype=torch.bool)
    positive_weights = positive_weights.masked_fill(mask_self, 0)

    weight_sum = positive_weights.sum(dim=1, keepdim=True) + 1e-8
    positive_weights = positive_weights / weight_sum

    exp_sim = torch.exp(sim_matrix)
    weighted_pos = (positive_weights * exp_sim).sum(dim=1)

    exp_sim = exp_sim.masked_fill(mask_self, 0)
    all_sum = exp_sim.sum(dim=1)

    return -torch.log(weighted_pos / (all_sum + 1e-8) + 1e-8).mean()
