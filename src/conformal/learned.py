"""Learned non-linear conformal scale: a small MLP trained by Adam on a conformal loss.

The scale sigma(x) divides the non-conformity score, ``s = (1 - p_y) / sigma(x)``, so a window
with a large sigma gets a looser threshold and a larger prediction set. ``train_scale`` fits
sigma from the four per-window signals by making the prediction-set construction smooth: a
quantile is taken on half of each batch, set membership on the other half is a sigmoid, and
the loss trades the mean soft set size against a per-region coverage shortfall.
"""

from __future__ import annotations

import copy
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

    Args:
        n_signals: number of input signals.
        hidden:    width of each hidden layer.
        depth:     number of hidden ReLU layers.

    Raises:
        ValueError: if ``n_signals`` is less than 1.
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
        """Map standardized signals to a strictly positive scale.

        The pre-activation is clamped to ``[-3, 3]`` before ``exp``, so sigma always lies
        in ``[e^-3, e^3]``.

        Args:
            signals_z: (n, n_signals) standardized signals.

        Returns:
            (n,) scale sigma.
        """
        f = self.net(signals_z).squeeze(-1)
        return torch.exp(torch.clamp(f, -_CLAMP, _CLAMP))


@dataclass
class TrainReport:
    """What training did: enough to tell "it learned nothing" from "it failed to run".

    Attributes:
        epochs:         epochs that ran (fewer than requested after an early stop).
        loss:           mean training loss per epoch.
        val_size:       soft |C| per epoch on the fit block's in-batch eval-half, not on
                        the held-out val block: the training curve, not a held-out
                        measurement.
        val_coverage:   soft coverage per epoch, measured like ``val_size``.
        scale_std:      the spread of sigma over the fit block at the end.
        earlystop_loss: the held-out early-stopping loss per epoch, or None when no
                        held-out block was given.
        best_epoch:     the 0-based epoch whose weights were restored, or None when
                        training ran without early stopping.
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

    Args:
        s_true:    (n,) scaled true-class scores of the batch.
        labels:    (n,) true regions of the batch.
        cal_i:     indices of the cal-half rows.
        n_classes: number of regions K.
        alpha:     miscoverage level.

    Returns:
        (K,) per-region quantiles.
    """
    q = s_true.new_full((n_classes,), 1e6)
    yc = labels[cal_i]
    for c in range(n_classes):
        m = yc == c
        if m.any():
            q[c] = torch.quantile(s_true[cal_i][m], 1.0 - alpha)
    return q


def _chronological_halves(groups: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    """Split rows into an earlier (cal) and a later (ev) half within each group.

    Rows are ordered by ``times`` inside each group. The first ``m // 2`` of a group's ``m`` rows
    go to cal, the rest to ev, so a group with a single row contributes to ev only.

    Args:
        groups: (n,) group id per row.
        times:  (n,) time order per row.

    Returns:
        ``(cal_i, ev_i)``: row indices into the inputs.
    """
    # Sort by time, then stably by group: rows end up grouped, time-ordered within a group.
    o = torch.argsort(times, stable=True)
    o = o[torch.argsort(groups[o], stable=True)]
    _, counts = torch.unique_consecutive(groups[o], return_counts=True)
    # Each sorted row's rank within its group: its position minus its group's start.
    starts = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    rank = torch.arange(len(o), device=o.device) - starts
    is_cal = rank < torch.repeat_interleave(counts // 2, counts)
    return o[is_cal], o[~is_cal]


def _in_batch_halves(
    n: int,
    split: str,
    gen: torch.Generator,
    groups: Tensor | None,
    times: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """(cal_i, ev_i) for one batch of ``n`` rows.

    ``random`` halves a random permutation. ``chronological`` puts each group's earlier half in
    cal and its later half in ev, so the quantile always comes from the past of the windows it is
    scored on, as it does at deployment.

    Args:
        n:      number of rows in the batch.
        split:  ``"random"`` or ``"chronological"``.
        gen:    generator for the random permutation.
        groups: (n,) group id per row; required by the chronological split.
        times:  (n,) time order per row; required by the chronological split.

    Returns:
        ``(cal_i, ev_i)``: row indices into the batch.

    Raises:
        ValueError: if the chronological split is missing ``groups`` or ``times``.
    """
    if split == "chronological":
        if groups is None or times is None:
            raise ValueError("the chronological split needs per-row groups and times")
        return _chronological_halves(groups, times)
    perm = torch.randperm(n, generator=gen, device=gen.device)
    half = n // 2
    return perm[:half], perm[half:]


def _conformal_loss_class_conditional(
    net: ScaleNet,
    signals: Tensor,
    probs: Tensor,
    labels: Tensor,
    n_classes: int,
    alpha: float,
    tau: float,
    kappa: float,
    cal_i: Tensor,
    ev_i: Tensor,
    size_weight: float = 1.0,
) -> tuple[Tensor, float, float]:
    """Conformal loss with per-region soft quantiles over a single shared scale.

    Each class is thresholded by its own region's soft quantile. The coverage shortfall is a mean
    over present regions. The scale is a single shared network, so gradients from every region's
    coverage term update the same weights: a per-region objective with pooled parameters. In-batch
    split trick: quantiles from the ``cal_i`` rows, coverage/size on the ``ev_i`` rows.

    ``loss = size_weight * mean soft |C| + kappa * mean over regions of max(0, (1 - alpha) -
    soft coverage)``, where a class's soft membership is ``sigmoid((q_c - s_c) / tau)``.

    Args:
        net:         the scale network being trained.
        signals:     (n, n_signals) standardized signals of the batch.
        probs:       (n, K) softmax probabilities.
        labels:      (n,) true regions.
        n_classes:   number of regions K.
        alpha:       miscoverage level.
        tau:         sigmoid temperature of the soft membership.
        kappa:       weight on the coverage shortfall.
        cal_i:       rows the quantiles come from.
        ev_i:        rows size and coverage are measured on.
        size_weight: weight on the size term.

    Returns:
        ``(loss, mean soft size, mean soft coverage)``; the last two are detached floats
        over the ev rows.
    """
    n = signals.shape[0]
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
    split: str = "random",
    groups: np.ndarray | None = None,
    val_groups: np.ndarray | None = None,
    init_net: ScaleNet | None = None,
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
        hidden:      width of each hidden layer of a new net.
        depth:       number of hidden layers of a new net.
        epochs:      maximum number of epochs.
        lr:          Adam learning rate.
        weight_decay: Adam L2 weight decay.
        batch_size:  rows per batch (capped at ``n``); a trailing partial batch is dropped.
        tau:         sigmoid temperature for soft set membership.
        kappa:       weight on the coverage shortfall.
        size_weight: weight on the sharpness term, relative to the coverage term.
        val_signals_z: (m, n_signals) standardized signals of the held-out early-stopping
                     block. Early stopping runs only when this, ``val_probs`` and
                     ``val_labels`` are all given.
        val_probs:   (m, n_classes) softmax probabilities of the held-out block.
        val_labels:  (m,) true class index of the held-out block.
        early_stop_patience: epochs without held-out improvement before stopping early
                     (None = never stop early, but still restore the best epoch).
        n_classes:   region count for the class-conditional loss, defaults to
                     ``probs.shape[1]``.
        split:       how each batch is halved into the quantile (cal) and scored (ev) rows.
                     ``"random"`` halves a random permutation. ``"chronological"`` puts the
                     earlier half of each group's rows in cal and the later half in ev.
        groups:      (n,) group id per fitting row (the session), needed by the chronological
                     split. Rows must be in chronological order within each group.
        val_groups:  the same for the early-stopping rows.
        init_net:    a trained net to continue from (fine-tuning); a copy is trained, the
                     argument is left untouched. ``None`` starts at sigma = 1.
        seed:        seed for the initialization, the batch order and the random split.
        device:      torch device string.

    Returns:
        (net, report). ``report.collapsed_to_standard`` is True when the net never left σ = 1,
        which is the honest "the signals earned nothing" outcome. ``report.best_epoch`` is the
        restored epoch when early stopping was used.

    Raises:
        ValueError: if the inputs differ in length, there are fewer than 32 fitting rows,
            ``split`` is unknown, or the chronological split is missing its groups.
    """
    if len(signals_z) != len(probs) or len(probs) != len(labels):
        raise ValueError(
            f"length mismatch: signals {len(signals_z)}, probs {len(probs)}, "
            f"labels {len(labels)}"
        )
    if len(labels) < 32:
        raise ValueError(f"need >= 32 fitting windows, got {len(labels)}")
    if split not in ("random", "chronological"):
        raise ValueError(f"split must be 'random' or 'chronological', got {split!r}")
    if split == "chronological" and (
        groups is None
        or len(groups) != len(labels)
        or (val_signals_z is not None and val_groups is None)
    ):
        raise ValueError(
            "the chronological split needs groups for every fitting row "
            "(and val_groups when early stopping)"
        )

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    dev = torch.device(device)

    X = torch.as_tensor(np.asarray(signals_z), dtype=torch.float32, device=dev)
    P = torch.as_tensor(np.asarray(probs), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=dev)

    if init_net is None:
        net = ScaleNet(X.shape[1], hidden=hidden, depth=depth).to(dev)
    else:
        net = copy.deepcopy(init_net).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    n_cls = n_classes if n_classes is not None else int(P.shape[1])
    G = None if groups is None else torch.as_tensor(np.asarray(groups), device=dev)

    def _loss(
        sig_b: Tensor, prob_b: Tensor, lab_b: Tensor, cal_i: Tensor, ev_i: Tensor
    ) -> tuple[Tensor, float, float]:
        """The conformal loss on one batch, with this run's fixed hyperparameters."""
        return _conformal_loss_class_conditional(
            net,
            sig_b,
            prob_b,
            lab_b,
            n_cls,
            alpha,
            tau,
            kappa,
            cal_i,
            ev_i,
            size_weight,
        )

    # optional held-out early-stopping block
    use_es = (
        val_signals_z is not None and val_probs is not None and val_labels is not None
    )
    if use_es:
        vX = torch.as_tensor(np.asarray(val_signals_z), dtype=torch.float32, device=dev)
        vP = torch.as_tensor(np.asarray(val_probs), dtype=torch.float32, device=dev)
        vY = torch.as_tensor(np.asarray(val_labels), dtype=torch.long, device=dev)
        # One fixed cal/ev split of the held-out block, drawn once, so the early-stopping loss
        # compares epochs on the same rows and does not advance the training generator.
        vG = (
            None
            if val_groups is None
            else torch.as_tensor(np.asarray(val_groups), device=dev)
        )
        v_cal, v_ev = _in_batch_halves(
            len(vY),
            split,
            torch.Generator(device=device).manual_seed(seed + 1),
            vG,
            torch.arange(len(vY), device=dev),
        )
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
            # The batch's row indices double as time order: the fitting rows are in
            # chronological order within each group.
            cal_i, ev_i = _in_batch_halves(
                len(idx), split, gen, None if G is None else G[idx], idx
            )
            loss, sz, cv = _loss(X[idx], P[idx], Y[idx], cal_i, ev_i)
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
                es_loss = float(_loss(vX, vP, vY, v_cal, v_ev)[0])
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
    """Evaluate a trained scale on standardized signals.

    Args:
        net:       the trained scale network; put in eval mode.
        signals_z: (n, n_signals) signals standardized with the net's training stats.
        device:    torch device string.

    Returns:
        (n,) strictly positive float64 σ.
    """
    net.eval()
    with torch.no_grad():
        X = torch.as_tensor(
            np.asarray(signals_z), dtype=torch.float32, device=torch.device(device)
        )
        return net(X).cpu().numpy().astype(np.float64)


def signal_sensitivity(
    net: ScaleNet,
    signals_raw: np.ndarray,
    signal_mean: np.ndarray,
    signal_std: np.ndarray,
) -> np.ndarray:
    """Per-window d(log sigma) / d(standardized signal), for each of the net's inputs.

    The relative change in sigma per standard deviation of a signal. Taking the log makes it
    independent of sigma's overall level, which carries no meaning (q_hat rescales with it), so
    values are comparable across mazes.

    Standardizes ``signals_raw`` (n, n_signals) with the net's own training mean/std, runs one
    forward pass, and backpropagates ``log(sigma).sum()`` to each input. Each window's sigma
    depends only on its own row, so the gradient of the sum is the per-window gradient.

    Args:
        net:         the trained scale network.
        signals_raw: (n, n_signals) raw signals in ``SIGNALS`` order.
        signal_mean: (n_signals,) the mean the net was trained with.
        signal_std:  (n_signals,) the std the net was trained with.

    Returns:
        (n, n_signals) gradients, one row per window.
    """
    net.eval()
    z = (np.asarray(signals_raw, dtype=float) - signal_mean) / signal_std
    z_t = torch.tensor(z, dtype=torch.float32, requires_grad=True)
    torch.log(net(z_t)).sum().backward()
    return z_t.grad.detach().numpy()


def signal_sensitivity_for_bundle(bundle: dict) -> dict[str, Any] | None:
    """Mean absolute ``signal_sensitivity`` over a chain bundle's test windows.

    Each tested session's sigma (``sigma_nets.pt``) is differentiated on that session's own
    test windows; the absolute gradients are then pooled over every test window.

    Args:
        bundle: a chain bundle from ``load_maze_bundles``, which records the paths of its
            ``sigma_nets.pt`` and ``arrays.npz``.

    Returns:
        ``{"signals": [...], "mean_abs_log_grad": array of shape (n_signals,)}``, or ``None`` if the
        bundle has no saved sigma nets or no saved arrays.
    """
    sn_path, arr_path = bundle.get("_sigma_nets_path"), bundle.get("_arrays_path")
    if sn_path is None or arr_path is None:
        return None
    payload = torch.load(sn_path, map_location="cpu", weights_only=False)
    names = list(payload["signals"])
    arrs = np.load(arr_path)
    raw = np.stack([arrs[f"win_{s}"] for s in names], axis=1)
    grads = []
    for i, sess in enumerate(payload["sessions"]):
        net = ScaleNet(len(names), hidden=payload["hidden"], depth=payload["depth"])
        net.load_state_dict(sess["state_dict"])
        rows = arrs["win_session_idx"] == i
        grads.append(
            signal_sensitivity(
                net,
                raw[rows],
                np.asarray(sess["signal_mean"]),
                np.asarray(sess["signal_std"]),
            )
        )
    return {
        "signals": names,
        "mean_abs_log_grad": np.abs(np.vstack(grads)).mean(axis=0),
    }
