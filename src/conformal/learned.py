"""Learned non-linear conformal scale: a small MLP trained by Adam on a conformal loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

# |f| is clamped to this before exp, so a diverging net cannot produce sigma = 0 or inf and
# silently empty every prediction set.
_CLAMP = 3.0


class ScaleNet(nn.Module):
    """A small MLP producing a strictly positive per-window scale.

    The output layer is zero-initialized, so at step 0 every window gets σ = 1 and the model
    is exactly standard split-conformal. Training can only depart from that baseline by
    reducing the loss.
    """

    def __init__(
        self,
        n_signals: int,
        hidden: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if n_signals < 1:
            raise ValueError(f"n_signals must be >= 1, got {n_signals}")
        layers: list[nn.Module] = []
        d = n_signals
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        head = nn.Linear(d, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)  # sigma = exp(0) = 1 -> starts AT standard conformal
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, signals_z: Tensor) -> Tensor:
        """(n, n_signals) standardized signals -> (n,) strictly positive scale."""
        f = self.net(signals_z).squeeze(-1)
        return torch.exp(torch.clamp(f, -_CLAMP, _CLAMP))


@dataclass
class TrainReport:
    """What training did? Enough to tell "it learned nothing" from "it failed to run".

    ``val_size`` and ``val_coverage`` are the soft |C| and soft coverage on the fit
    block's in-batch eval-half, not on the held-out val block. They are the training
    curve, not a held-out measurement. ``earlystop_loss`` is the held-out one.

    ``scale_std`` is the spread of sigma at the end. ``best_epoch`` is the epoch whose
    weights were restored, or None when training ran to the end.
    """

    epochs: int
    loss: list[float]
    val_size: list[float]
    val_coverage: list[float]
    scale_std: float
    earlystop_loss: list[float] | None = None
    best_epoch: int | None = None

    @property
    def collapsed_to_standard(self) -> bool:
        """True if the net never moved away from sigma = 1 (learned nothing)."""
        return self.scale_std < 1e-3


def _class_conditional_soft_quantiles(
    s_true: Tensor, labels: Tensor, cal_i: Tensor, n_classes: int, alpha: float
) -> Tensor:
    """Per-region soft (1-alpha) quantile from the cal-half.

    ``torch.quantile`` is differentiable through the order statistic it selects, so gradients flow
    to the shared scale. A region absent from the cal-half gets a large sentinel -> its class is
    freely admitted on the ev-half that step (no gradient), the soft analog of the +inf
    calibration fallback.
    """
    q = s_true.new_full((n_classes,), 1e6)
    yc = labels[cal_i]
    for c in range(n_classes):
        m = yc == c
        if m.any():
            q[c] = torch.quantile(s_true[cal_i][m], 1.0 - alpha)
    return q


def _conformal_loss_class_conditional(
    net: ScaleNet,
    signals: Tensor,
    probs: Tensor,
    labels: Tensor,
    n_classes: int,
    alpha: float,
    tau: float,
    kappa: float,
    gen: torch.Generator,
    size_weight: float = 1.0,
) -> tuple[Tensor, float, float]:
    """Conformal loss with per-region soft quantiles over a single shared scale.

    Each class is thresholded by its own region's soft quantile. The coverage shortfall is a mean
    over present regions. The scale is a single shared network, so gradients from every region's
    coverage term update the same weights: a per-region objective with pooled parameters. In-batch
    split trick: quantiles from one half, coverage/size on the other.
    """
    n = signals.shape[0]
    perm = torch.randperm(n, generator=gen, device=signals.device)
    half = n // 2
    cal_i, ev_i = perm[:half], perm[half:]

    scale = net(signals)  # (n,) shared
    ar = torch.arange(n, device=probs.device)
    s_all = (1.0 - probs) / scale.unsqueeze(1)  # (n, K)
    s_true = (1.0 - probs[ar, labels]) / scale  # (n,)

    q = _class_conditional_soft_quantiles(
        s_true, labels, cal_i, n_classes, alpha
    )  # (K,)
    member = torch.sigmoid((q.unsqueeze(0) - s_all[ev_i]) / tau)  # (n_ev, K)
    size = member.sum(dim=1)
    y_ev = labels[ev_i]
    cover = member[torch.arange(len(ev_i), device=probs.device), y_ev]

    shorts, npres = size.new_zeros(()), 0
    for c in range(n_classes):
        m = y_ev == c
        if m.any():
            shorts = shorts + torch.clamp((1.0 - alpha) - cover[m].mean(), min=0.0)
            npres += 1
    # balanced over present regions; size_weight < 1 is more liberal toward larger sets
    loss = size_weight * size.mean() + kappa * shorts / max(npres, 1)
    return loss, float(size.mean().detach()), float(cover.mean().detach())


def train_scale(
    signals_z: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
    hidden: int = 32,
    depth: int = 2,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    tau: float = 0.05,
    kappa: float = 20.0,
    size_weight: float = 1.0,
    val_signals_z: np.ndarray | None = None,
    val_probs: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    early_stop_patience: int | None = None,
    n_classes: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[ScaleNet, TrainReport]:
    """Fit σ_θ by Adam on the per-region conformal loss.

    Trains under the class-conditional loss (``_conformal_loss_class_conditional``): per-region
    soft quantiles, coverage shortfall averaged over present regions, a single shared scale.

    Args:
        signals_z:   (n, n_signals) standardized signals for the fitting block.
        probs:       (n, n_classes) softmax probabilities for the same windows.
        labels:      (n,) true class index.
        alpha:       miscoverage level, coverage target is 1 - alpha.
        tau:         sigmoid temperature for soft set membership.
        kappa:       weight on the coverage shortfall.
        size_weight: weight on the sharpness term, relative to the coverage term.
        early_stop_patience: epochs without held-out improvement before stopping early.
        n_classes:   region count for the class-conditional loss, defaults to
                     ``probs.shape[1]``.

    Returns:
        (net, report). ``report.collapsed_to_standard`` is True when the net never left σ = 1,
        which is the honest "the signals earned nothing" outcome. ``report.best_epoch`` is the
        restored epoch when early stopping was used.
    """
    if len(signals_z) != len(probs) or len(probs) != len(labels):
        raise ValueError(
            f"length mismatch: signals {len(signals_z)}, probs {len(probs)}, "
            f"labels {len(labels)}"
        )
    if len(labels) < 32:
        raise ValueError(f"need >= 32 fitting windows, got {len(labels)}")

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    dev = torch.device(device)

    X = torch.as_tensor(np.asarray(signals_z), dtype=torch.float32, device=dev)
    P = torch.as_tensor(np.asarray(probs), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)

    net = ScaleNet(X.shape[1], hidden=hidden, depth=depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    n_cls = n_classes if n_classes is not None else int(P.shape[1])

    def _batch_loss(
        sig_b: Tensor, prob_b: Tensor, lab_b: Tensor
    ) -> tuple[Tensor, float, float]:
        return _conformal_loss_class_conditional(
            net, sig_b, prob_b, lab_b, n_cls, alpha, tau, kappa, gen, size_weight
        )

    # optional held-out early-stopping block
    use_es = (
        val_signals_z is not None and val_probs is not None and val_labels is not None
    )
    if use_es:
        vX = torch.as_tensor(np.asarray(val_signals_z), dtype=torch.float32, device=dev)
        vP = torch.as_tensor(np.asarray(val_probs), dtype=torch.float32, device=dev)
        vY = torch.as_tensor(np.asarray(val_labels), dtype=torch.long, device=dev)
    patience = epochs if early_stop_patience is None else int(early_stop_patience)

    n = len(Y)
    bs = min(batch_size, n)
    hist_loss: list[float] = []
    hist_size: list[float] = []
    hist_cov: list[float] = []
    hist_es: list[float] = []
    best_es, best_epoch, best_state, since_best = np.inf, None, None, 0

    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen, device=dev)
        ep_loss = ep_size = ep_cov = 0.0
        nb = 0
        for lo in range(0, n - bs + 1, bs):
            idx = perm[lo : lo + bs]
            loss, sz, cv = _batch_loss(X[idx], P[idx], Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_size += sz
            ep_cov += cv
            nb += 1
        if nb:
            hist_loss.append(ep_loss / nb)
            hist_size.append(ep_size / nb)
            hist_cov.append(ep_cov / nb)

        if use_es:
            with torch.no_grad():
                es_loss = float(_batch_loss(vX, vP, vY)[0])
            hist_es.append(es_loss)
            if es_loss < best_es:
                best_es, best_epoch, since_best = es_loss, ep, 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
            else:
                since_best += 1
                if since_best >= patience:
                    break

    if use_es and best_state is not None:
        net.load_state_dict(best_state)  # best-checkpoint restore

    with torch.no_grad():
        spread = float(net(X).std())
    return net, TrainReport(
        epochs=len(hist_loss),
        loss=hist_loss,
        val_size=hist_size,
        val_coverage=hist_cov,
        scale_std=spread,
        earlystop_loss=hist_es if use_es else None,
        best_epoch=best_epoch,
    )


def scale_of(net: ScaleNet, signals_z: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(n,) strictly positive σ for the given standardized signals."""
    net.eval()
    with torch.no_grad():
        X = torch.as_tensor(
            np.asarray(signals_z), dtype=torch.float32, device=torch.device(device)
        )
        return net(X).cpu().numpy().astype(np.float64)


def load_scale_net(
    checkpoint: dict,
) -> tuple[ScaleNet, list[str], np.ndarray, np.ndarray]:
    """Reconstruct a trained ``ScaleNet`` from the checkpoint dict ``per_maze.ipynb`` saves.

    Returns:
        ``(net, signals, signal_mean, signal_std)``: ``net`` in eval mode, ``signals`` is the
        input-column order the checkpoint was trained on, ``signal_mean``/``signal_std`` are the
        exact standardization stats ``ũ(x)`` used at training time, so a caller can rebuild the
        model's inputs from raw per-window signal values without re-deriving them.
    """
    signals = list(checkpoint["signals"])
    net = ScaleNet(
        n_signals=len(signals),
        hidden=int(checkpoint["hidden"]),
        depth=int(checkpoint["depth"]),
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    mean = np.asarray(checkpoint["signal_mean"], dtype=float)
    std = np.asarray(checkpoint["signal_std"], dtype=float)
    return net, signals, mean, std


def signal_sensitivity(
    net: ScaleNet,
    signals_raw: np.ndarray,
    signal_mean: np.ndarray,
    signal_std: np.ndarray,
) -> np.ndarray:
    """Per-window d(sigma) / d(standardized signal), for each of the net's inputs.

    Standardizes ``signals_raw`` (n, n_signals) with the net's own training mean/std, runs one
    forward pass, and backpropagates ``sigma.sum()`` to get sigma's gradient with respect to
    each input.

    Returns:
        (n, n_signals) gradients, one row per window.
    """
    net.eval()
    z = (np.asarray(signals_raw, dtype=float) - signal_mean) / signal_std
    z_t = torch.tensor(z, dtype=torch.float32, requires_grad=True)
    sigma = net(z_t)
    sigma.sum().backward()
    return z_t.grad.detach().numpy()


def signal_sensitivity_for_bundle(bundle: dict) -> dict[str, Any] | None:
    """Mean absolute ``signal_sensitivity``, computed from a pooled bundle's saved files.

    Returns:
        ``{"signals": [...], "mean_abs_grad": array of shape (n_signals,)}``, or ``None`` if the
        bundle has no saved scale net or no saved arrays.
    """
    sn_path, arr_path = bundle.get("_scale_net_path"), bundle.get("_arrays_path")
    if sn_path is None or arr_path is None:
        return None

    ckpt = torch.load(sn_path, map_location="cpu", weights_only=False)
    net, signal_names, sig_mean, sig_std = load_scale_net(ckpt)
    arrs = np.load(arr_path)
    test_mask = arrs["win_split"] == 2
    raw = np.stack([arrs[f"win_{s}"][test_mask] for s in signal_names], axis=1)
    grad = signal_sensitivity(net, raw, sig_mean, sig_std)
    return {"signals": signal_names, "mean_abs_grad": np.abs(grad).mean(axis=0)}
