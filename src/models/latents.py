"""Latent extraction utilities for Step 1 and Step 2 inference.

Every extractor puts its model in eval mode, runs without gradients in batches, and returns
one numpy row per input window.
"""

import numpy as np
import torch

from .alignment import AlignmentDecoder
from .encoder import SharedTemporalClassifier


def extract_latents_hybrid(
    model: SharedTemporalClassifier,
    windows: np.ndarray,
    session_name: str,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract Step 1 latents (mu) for all windows of a session.

    Args:
        model:        SharedTemporalClassifier
        windows:      spike count windows, shape (n_windows, T, n_neurons)
        session_name: session key used by the encoder's per-session weights
        device:       compute device
        batch_size:   number of windows per forward pass

    Returns:
        latents: (n_windows, latent_dim) float32 numpy array
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(
                windows[i : i + batch_size], dtype=torch.float32, device=device
            )
            z = model.get_latent(batch, session_name)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def extract_logvar(
    model: SharedTemporalClassifier,
    windows: np.ndarray,
    session_name: str,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract the BC-VAE posterior log-variance (mu's uncertainty) per window.

    Args:
        model:        SharedTemporalClassifier
        windows:      spike count windows, shape (n_windows, T, n_neurons)
        session_name: session key used by the encoder's per-session weights
        device:       compute device
        batch_size:   number of windows per forward pass

    Returns:
        logvar: (n_windows, latent_dim) float numpy array
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            x = torch.tensor(
                windows[i : i + batch_size], dtype=torch.float32, device=device
            )
            _, lv = model.get_latent_and_logvar(x, session_name)
            out.append(lv.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_latent_sequences(all_latents: np.ndarray, seq_len: int) -> np.ndarray:
    """Build sliding-window sequences of latents for Step 2.

    Pads the beginning by repeating the first latent (seq_len - 1) times so
    every window has a full context window.

    Args:
        all_latents: (n_windows, latent_dim)
        seq_len:     sliding window length (STEP2_SEQ_LEN=5)

    Returns:
        sequences: (n_windows, seq_len, latent_dim)
    """
    padded = np.concatenate(
        [np.tile(all_latents[0:1], (seq_len - 1, 1)), all_latents], axis=0
    )
    n = len(all_latents)
    return np.array([padded[i : i + seq_len] for i in range(n)], dtype=np.float32)


def extract_aligned_latents(
    alignment_decoder: AlignmentDecoder,
    all_latents: np.ndarray,
    device: torch.device,
    seq_len: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract Step 2 aligned latents for all windows of a session.

    Args:
        alignment_decoder: AlignmentDecoder (trained)
        all_latents:       Step 1 latents, shape (n_windows, latent_dim)
        device:            compute device
        seq_len:           sliding window length (STEP2_SEQ_LEN=5)
        batch_size:        sequences per forward pass

    Returns:
        aligned: (n_windows, latent_dim) float32 numpy array
    """
    alignment_decoder.eval()
    sequences = build_latent_sequences(all_latents, seq_len)
    aligned = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[i : i + batch_size]).to(device)
            z_aligned = alignment_decoder.align(batch)
            aligned.append(z_aligned.cpu().numpy())
    return np.concatenate(aligned, axis=0)
