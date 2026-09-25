"""Jacobian strain computation for the AlignmentDecoder alignment map.

For each latent sequence z_seq = [z_{t-k}, ..., z_t], the alignment strain is:

    J_t(z) = d f_align(z_seq) / d z_t     shape: (latent_dim, latent_dim)
    ||J_t(z)||_2 = sigma_max(J_t(z))      the ``align_jacobian`` signal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..models.alignment import AlignmentDecoder
    from ..models.encoder import SharedTemporalClassifier


def _top_singular_value(
    outputs: torch.Tensor, inputs: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Largest singular value of d outputs / d inputs, per row of the batch."""
    b, out_dim = outputs.shape
    jac = torch.zeros(b, out_dim, inputs[0].numel(), device=device)
    for j in range(out_dim):
        (grad,) = torch.autograd.grad(
            outputs[:, j].sum(),
            inputs,
            retain_graph=(j < out_dim - 1),
            create_graph=False,
        )
        jac[:, j, :] = grad.reshape(b, -1)
    # svdvals returns descending order
    return torch.linalg.svdvals(jac)[:, 0].detach().cpu().numpy()


def compute_spectral_norms(
    model: AlignmentDecoder,
    z_seqs: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute the Jacobian spectral norm for each sequence in a batch.

    For each sequence the Jacobian of z_aligned with respect to the last input is
    computed, then the spectral norm (largest singular value) is extracted.

    Args:
        model:      trained AlignmentDecoder, put in eval mode internally
        z_seqs:     (N, seq_len, latent_dim) numpy array or float32 tensor
        batch_size: sequences processed in one backward-pass group

    Returns:
        spectral_norms: (N,) float32 numpy array, one value per sequence
    """
    if isinstance(z_seqs, np.ndarray):
        z_seqs = torch.from_numpy(z_seqs).float()

    model.eval()
    n = z_seqs.shape[0]
    spectral_norms = np.zeros(n, dtype=np.float32)

    with torch.enable_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = z_seqs[start:end].to(device)  # (B, seq_len, latent_dim)

            z_context = batch[:, :-1, :].detach()  # (B, seq_len-1, latent_dim)
            z_last = batch[:, -1, :].detach().requires_grad_(True)  # (B, latent_dim)

            z_seq_mod = torch.cat([z_context, z_last.unsqueeze(1)], dim=1)
            z_aligned = model.align(z_seq_mod)  # (B, latent_dim)

            spectral_norms[start:end] = _top_singular_value(z_aligned, z_last, device)

    return spectral_norms


def compute_encoder_spectral_norms(
    encoder: SharedTemporalClassifier,
    windows: np.ndarray | torch.Tensor,
    session_name: str,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-window spectral norm of the BC-VAE encoder Jacobian d mu / d x.

    For each input window x (window_size x n_neurons), differentiates the posterior
    mean mu = encoder.get_latent(x, session_name) with respect to the flattened input
    and returns the largest singular value of the Jacobian.

    Args:
        encoder:      trained BC-VAE encoder, set to eval mode internally.
        windows:      (N, window_size, n_neurons) float32 array or tensor (the same
                      normalized windows the encoder consumes).
        session_name: session key for the encoder's per-session projection.
        batch_size:   windows processed in one backward-pass group.

    Returns:
        spectral_norms: (N,) float32 numpy array, one value per window.
    """
    if isinstance(windows, np.ndarray):
        windows = torch.from_numpy(windows).float()

    encoder.eval()
    n = windows.shape[0]
    spectral_norms = np.zeros(n, dtype=np.float32)

    with torch.enable_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = windows[start:end].to(device).clone().requires_grad_(True)  # (B, T, C)
            mu = encoder.get_latent(x, session_name)  # (B, latent_dim)

            # the Jacobian is taken against the flattened window
            spectral_norms[start:end] = _top_singular_value(mu, x, device)

    return spectral_norms
