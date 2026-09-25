"""BC-VAE encoder model for cross-session neural population embedding."""

import torch
import torch.nn as nn

from .config import FACTOR_DIM, HIDDEN_DIM, LATENT_DIM, SHARED_DIM


class _SharedSessionEncoder(nn.Module):
    """Shared VAE encoder.

    Neurons are split into shared (present in all sessions) and session-specific.
    """

    def __init__(
        self,
        n_shared: int,
        session_configs: dict,
        shared_dim: int = SHARED_DIM,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        for session_name, cfg in session_configs.items():
            self.register_buffer(
                f"shared_idx_{session_name}",
                torch.tensor(cfg["shared_indices"], dtype=torch.long),
            )
            self.register_buffer(
                f"specific_idx_{session_name}",
                torch.tensor(cfg["specific_indices"], dtype=torch.long),
            )

        self.shared_projection = nn.Sequential(
            nn.Linear(n_shared, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
        )
        self.session_projections: nn.ModuleDict = nn.ModuleDict()
        for session_name, cfg in session_configs.items():
            if cfg["n_specific"] > 0:
                self.session_projections[session_name] = nn.Sequential(
                    nn.Linear(cfg["n_specific"], shared_dim),
                    nn.LayerNorm(shared_dim),
                    nn.ReLU(),
                )

        self.fusion_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim), nn.Sigmoid()
        )
        self.fusion_transform = nn.Linear(shared_dim * 2, shared_dim)

        self.temporal_encoder = nn.GRU(
            shared_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self, x: torch.Tensor, session_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, T, _ = x.shape
        shared_idx = getattr(self, f"shared_idx_{session_name}")
        specific_idx = getattr(self, f"specific_idx_{session_name}")

        x_shared = x[:, :, shared_idx].reshape(batch * T, -1)
        h_shared = self.shared_projection(x_shared)

        if len(specific_idx) > 0 and session_name in self.session_projections:
            x_spec = x[:, :, specific_idx].reshape(batch * T, -1)
            h_spec = self.session_projections[session_name](x_spec)
        else:
            h_spec = torch.zeros_like(h_shared)

        h_cat = torch.cat([h_shared, h_spec], dim=-1)
        gate = self.fusion_gate(h_cat)
        h = gate * h_shared + (1 - gate) * self.fusion_transform(h_cat)
        h = h.reshape(batch, T, -1)

        _, h_n = self.temporal_encoder(h)
        h_final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc_mu(h_final), self.fc_logvar(h_final)


class SharedTemporalClassifier(nn.Module):
    """BC-VAE.

    Preferred for cross-session generalization: the shared parameters see the
    same feature space across all sessions.

    Args:
        n_shared: number of neurons present in all sessions
        session_configs: dict mapping session_name →
            {'n_specific': int, 'shared_indices': list[int],
             'specific_indices': list[int], ...}
        n_classes: number of region labels (8 for R0-R7)
    """

    def __init__(
        self,
        n_shared: int,
        session_configs: dict,
        n_classes: int,
        shared_dim: int = SHARED_DIM,
        latent_dim: int = LATENT_DIM,
        factor_dim: int = FACTOR_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.encoder = _SharedSessionEncoder(
            n_shared, session_configs, shared_dim, latent_dim, hidden_dim
        )
        self.fc_g0 = nn.Linear(latent_dim, hidden_dim)
        self.generator = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.fc_factors = nn.Linear(hidden_dim, factor_dim)
        self.session_read_outs = nn.ModuleDict(
            {
                s: nn.Sequential(nn.Linear(factor_dim, cfg["n_total"]), nn.Softplus())
                for s, cfg in session_configs.items()
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw z -> N(mu, exp(logvar)).

        Args:
            mu:     (batch, latent_dim) posterior mean
            logvar: (batch, latent_dim) posterior log-variance

        Returns:
            (batch, latent_dim) sampled latent, differentiable in mu and logvar.
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def generate(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """Unroll a latent into T per-bin generator factors.

        Args:
            z: (batch, latent_dim) latent, sampled or deterministic
            T: number of time bins to unroll (``WINDOW_SIZE`` for a full window)

        Returns:
            (batch, T, factor_dim) factors; a session read-out turns these into the
            per-neuron Poisson rates ``train_step1`` reconstructs.
        """
        g0 = self.fc_g0(z).unsqueeze(0)
        gen_input = z.unsqueeze(1).expand(-1, T, -1)
        gen_output, _ = self.generator(gen_input, g0)
        return self.fc_factors(gen_output)

    def forward(
        self, x: torch.Tensor, session_name: str, return_all: bool = False
    ) -> torch.Tensor | dict:
        """Encode one session's windows, then decode them back to Poisson rates.

        Args:
            x:            (batch, T, n_total) window of spike counts for this session,
                          columns laid out per ``session_configs[session_name]``
            session_name: selects the per-session input projection and read-out
            return_all:   False returns the sampled latent alone, True returns every
                          tensor ``train_step1`` needs to form its three loss terms

        Returns:
            The sampled latent ``z`` of shape (batch, latent_dim) when ``return_all``
            is False, otherwise a dict with keys ``z``, ``mu``, ``logvar``, ``factors``,
            ``rates`` (Poisson target) and ``region_logits``.
        """
        _batch, T, _ = x.shape
        mu, logvar = self.encoder(x, session_name)
        z = self.reparameterize(mu, logvar)
        factors = self.generate(z, T)
        # Classify from the sampled latent z during training.
        region_logits = self.classifier(z if self.training else mu)

        if return_all:
            rates = self.session_read_outs[session_name](factors)
            return {
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "factors": factors,
                "rates": rates,
                "region_logits": region_logits,
            }
        return z

    def get_latent(self, x: torch.Tensor, session_name: str) -> torch.Tensor:
        """Return mu only."""
        mu, _ = self.encoder(x, session_name)
        return mu

    def get_latent_and_logvar(
        self, x: torch.Tensor, session_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for use as conformal uncertainty signal."""
        return self.encoder(x, session_name)
