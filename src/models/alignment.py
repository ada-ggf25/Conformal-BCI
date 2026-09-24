"""Transformer alignment decoder for cross-session latent alignment.

Adapted from the Liu Lab's cross-session alignment codebase.

Takes a sliding window of seq_len consecutive BC-VAE latents (z_step1) from
potentially different sessions and produces:

  z_aligned  — aligned latent (dim=latent_dim), input to the conformal layer
  proj       — L2-normalized projection (dim=64), used in NT-Xent contrastive loss
  logits     — region class predictions (dim=n_classes), input to conformal calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HIDDEN_DIM, LATENT_DIM, STEP2_SEQ_LEN


class AlignmentDecoder(nn.Module):
    """Transformer encoder that aligns a sequence of latents across sessions.

    Args:
        latent_dim: BC-VAE latent dimension (default: LATENT_DIM=16)
        n_classes:  number of region labels (8 for R0-R7)
        seq_len:    sliding window length in latent steps (default: STEP2_SEQ_LEN=5)
        n_heads:    Transformer attention heads
        n_layers:   Transformer encoder layers
        d_model:    Transformer model width
        dropout:    Transformer dropout rate.
        hidden_dim: classifier MLP hidden width
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        n_classes: int = 8,
        seq_len: int = STEP2_SEQ_LEN,
        n_heads: int = 4,
        n_layers: int = 2,
        d_model: int = 64,
        dropout: float = 0.1,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

        # Projection head for NT-Xent contrastive loss (Step 2 training only)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Region classifier head
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def align(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Map a sequence of latents to a single aligned latent via Transformer.

        Args:
            z_seq: (batch, seq_len, latent_dim)

        Returns:
            z_aligned: (batch, latent_dim) — last-token output
        """
        h = self.input_proj(z_seq) + self.pos_embed
        h = self.transformer(h)
        return self.output_proj(h[:, -1, :])

    def forward(
        self, z_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            z_seq: (batch, seq_len, latent_dim): consecutive z_step1 latents

        Returns:
            z_aligned: (batch, latent_dim)
            proj:      (batch, 64) — L2-normalized, for NT-Xent loss
            logits:    (batch, n_classes) — region class logits, softmax → conformal
        """
        z_aligned = self.align(z_seq)
        proj = F.normalize(self.projector(z_aligned), dim=-1)
        logits = self.decoder(z_aligned)
        return z_aligned, proj, logits
