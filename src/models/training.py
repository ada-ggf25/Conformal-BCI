"""Training loops for Step 1 (BC-VAE encoder) and Step 2 (Transformer alignment decoder)."""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    KL_WARMUP_EPOCHS,
    KL_WEIGHT,
    MAIN_EARLY_STOP_MIN_DELTA,
    MAIN_EARLY_STOP_PATIENCE,
    RECON_WEIGHT,
    REGION_WEIGHT,
    TEMPERATURE,
)
from .losses import compute_contrastive_loss, poisson_nll


def _weighted_ce_sums(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> tuple[float, float]:
    """Numerator and denominator of a class-weighted cross-entropy mean over one batch.

    Summed over a whole validation block, their ratio is the weighted mean over every
    window. A per-batch mean is 0/0 = NaN for a batch whose windows all belong to
    zero-weight classes (regions absent from the training block), and one such batch would
    make the averaged validation loss NaN.

    Args:
        logits:  (B, K) class logits.
        targets: (B,) class indices.
        weights: (K,) class weights.

    Returns:
        ``(sum_i w_{y_i} * nll_i, sum_i w_{y_i})``.
    """
    num = nn.functional.cross_entropy(logits, targets, weight=weights, reduction="sum")
    return float(num.item()), float(weights[targets].sum().item())


def _weighted_mean(num: float, den: float) -> float:
    """``num / den``, or NaN when every validation window has zero weight."""
    return num / den if den > 0 else float("nan")


def _es_defaults(patience, min_delta):
    """Early-stopping patience and minimum delta, each falling back to its config default.

    Args:
        patience:  epochs without improvement before stopping, or None.
        min_delta: minimum improvement that resets the patience counter, or None.

    Returns:
        ``(patience, min_delta)``, with None replaced by ``MAIN_EARLY_STOP_PATIENCE`` and
        ``MAIN_EARLY_STOP_MIN_DELTA``.
    """
    p = MAIN_EARLY_STOP_PATIENCE if patience is None else patience
    d = MAIN_EARLY_STOP_MIN_DELTA if min_delta is None else min_delta
    return p, d


def _step1_param_groups(
    model: nn.Module, weight_decay: float, classifier_weight_decay: float
) -> list[dict]:
    """Adam param groups that regularize the classifier head separately.

    Args:
        model:                   the model to optimize; its ``classifier`` attribute, if
                                 any, is the head.
        weight_decay:            decay on the other weight matrices.
        classifier_weight_decay: decay on every classifier-head parameter.

    Returns:
        Up to three param groups, empty groups omitted: weight matrices outside the head
        (``weight_decay``), biases and LayerNorm weights outside the head (no decay), and
        the whole head (``classifier_weight_decay``).
    """
    classifier = getattr(model, "classifier", None)
    clf_ids = (
        {id(p) for p in classifier.parameters()} if classifier is not None else set()
    )
    decay: list = []
    no_decay: list = []
    clf: list = []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in clf_ids:
            clf.append(p)
        elif p.ndim <= 1:  # LayerNorm weights, biases
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": clf, "weight_decay": classifier_weight_decay},
    ]
    return [g for g in groups if g["params"]]


def _inverse_frequency_weights(
    class_counts: dict | None,
    n_classes: int,
    power: float,
    device: torch.device,
    log_fn: Callable[[str], None] | None = None,
) -> torch.Tensor:
    """Inverse-frequency CrossEntropy weights: w_c = (total / (K * count_c)) ** power.

    The weights are then rescaled to sum to K, so the loss keeps its scale. A class absent
    from ``class_counts`` gets weight 0, and a class with a zero count gets 1 before the
    rescaling.

    Args:
        class_counts: ``{class_idx: count}`` over the training windows, or None for
                      uniform weights.
        n_classes:    K, the number of classes.
        power:        exponent on the inverse frequency (0 = uniform, 1 = full).
        device:       device of the returned tensor.
        log_fn:       optional sink for a one-line summary of the weights.

    Returns:
        (K,) float tensor of class weights.
    """
    if class_counts is None:
        return torch.ones(n_classes, device=device)
    total = sum(class_counts.values())
    weights = torch.zeros(n_classes, device=device)
    for cls, count in class_counts.items():
        if cls < n_classes:
            inv_freq = total / (n_classes * count) if count > 0 else 1.0
            weights[cls] = inv_freq**power
    weights = weights / weights.sum() * n_classes
    if log_fn is not None:
        log_fn(
            f"    Class weights (power={power}): "
            f"{[f'{w:.2f}' for w in weights.tolist()]}"
        )
    return weights


def train_step1[M: nn.Module](
    model: M,
    loader: DataLoader,
    device: torch.device,
    log_fn: Callable[[str], None],
    n_classes: int,
    num_epochs: int,
    lr: float,
    class_counts: dict | None = None,
    val_loader: DataLoader | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    weight_decay: float = 0.0,
    classifier_weight_decay: float = 0.0,
    region_weight: float = REGION_WEIGHT,
    class_weight_power: float = 1.0,
    kl_weight: float = KL_WEIGHT,
    kl_free_bits: float = 0.0,
    kl_warmup_epochs: int = KL_WARMUP_EPOCHS,
    log_every: int = 50,
) -> tuple[M, dict]:
    """Train the BC-VAE encoder (Step 1).

    Loss = RECON_WEIGHT * Poisson_NLL + region_weight * CrossEntropy + kl_w * KL,
    where ``kl_w`` ramps ``kl_weight`` linearly over the first ``kl_warmup_epochs`` epochs.

    Early stopping ignores the warm-up: the validation loss carries the same ramping KL weight,
    so a warm-up epoch looks better only because its KL term is under-weighted. No checkpoint
    is taken and no patience is spent until ``kl_w`` reaches ``kl_weight``.

    Args:
        model:          SharedTemporalClassifier; trained in place and returned.
        loader:         DataLoader over a HybridClassificationDataset, collated by
                        ``hybrid_collate_fn`` into per-session batch dicts.
        device:         torch.device
        log_fn:         callable(str) for progress output
        n_classes:      number of region labels
        num_epochs:     training epochs; also the cosine schedule's period.
        lr:             AdamW learning rate (STEP1_LEARNING_RATE=1e-3)
        class_counts:   optional dict {class_idx: count} for inverse-frequency weighting
        val_loader:     optional DataLoader for validation loss
        early_stop_patience: epochs without val_loss improvement before stopping
                        (None = ``MAIN_EARLY_STOP_PATIENCE``).
        early_stop_min_delta: minimum improvement to reset patience counter
                        (None = ``MAIN_EARLY_STOP_MIN_DELTA``).
        weight_decay:   AdamW L2 decay on the encoder/generator (0.0 = none)
        classifier_weight_decay: AdamW L2 decay applied to the region-classifier head only
        region_weight:  weight on the region CrossEntropy term.
        class_weight_power: exponent applied to inverse-frequency class weights.
        kl_weight:      weight on the KL term.
        kl_free_bits:   per-dimension KL floor in nats (0.0 = off).
        kl_warmup_epochs: length of the linear KL ramp, also the first epoch early
                        stopping may select (0 = no ramp).
        log_every:      print a progress line every this many epochs (default 50)

    Returns:
        (model, history_dict) where history_dict contains per-epoch lists:
        epoch, loss, recon, region, kl, kl_weight, acc, val_loss, val_recon,
        val_region and val_kl, plus ``best_epoch``: the 1-based epoch whose weights
        were restored, or None when none were.
    """
    optimizer = torch.optim.AdamW(
        _step1_param_groups(model, weight_decay, classifier_weight_decay), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    es_patience, es_delta = _es_defaults(early_stop_patience, early_stop_min_delta)

    class_weights = _inverse_frequency_weights(
        class_counts, n_classes, class_weight_power, device, log_fn
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log_fn("    Training Step 1 (Recon + Region + KL, NO contrastive)...")

    hist: dict = {}
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch: int | None = None
    stall = 0

    for epoch in range(num_epochs):
        warmed_up = epoch + 1 >= kl_warmup_epochs
        kl_w = kl_weight if warmed_up else kl_weight * (epoch + 1) / kl_warmup_epochs
        model.train()
        total_loss = total_recon = total_region = total_kl = 0.0
        correct = total = n_batches = 0

        for batch_dict in loader:
            optimizer.zero_grad()
            all_mu, all_logvar, all_logits, all_regions = [], [], [], []
            session_recons: list[torch.Tensor] = []

            for session_name, session_batch in batch_dict.items():
                windows = session_batch["windows"].to(device)
                # Poisson NLL target = raw spike counts (non-negative), not the
                # z-scored encoder input.
                recon_target = session_batch["windows_raw"].to(device)
                regions = session_batch["regions"].to(device)
                out = model(windows, session_name, return_all=True)
                all_mu.append(out["mu"])
                all_logvar.append(out["logvar"])
                all_logits.append(out["region_logits"])
                all_regions.append(regions)
                session_recons.append(poisson_nll(out["rates"], recon_target))

            mu = torch.cat(all_mu)
            logvar = torch.cat(all_logvar)
            logits = torch.cat(all_logits)
            regions = torch.cat(all_regions)

            recon_loss = torch.stack(session_recons).mean()
            # Per-dim KL (mean over batch), optional free-bits floor prevents any
            # single latent dimension's KL from being driven below kl_free_bits
            # nats. With kl_free_bits=0 this is identical to the plain mean KL.
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
            if kl_free_bits > 0.0:
                kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
            kl_loss = kl_per_dim.mean()
            region_loss = criterion(logits, regions)
            loss = (
                RECON_WEIGHT * recon_loss + region_weight * region_loss + kl_w * kl_loss
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_region += region_loss.item()
            total_kl += kl_loss.item()
            _, predicted = torch.max(logits, 1)
            total += regions.size(0)
            correct += int((predicted == regions).sum().item())
            n_batches += 1

        scheduler.step()
        n_batches = max(n_batches, 1)

        epoch_acc = 100.0 * correct / max(total, 1)
        hist.setdefault("epoch", []).append(epoch + 1)
        hist.setdefault("loss", []).append(total_loss / n_batches)
        hist.setdefault("recon", []).append(total_recon / n_batches)
        hist.setdefault("region", []).append(total_region / n_batches)
        hist.setdefault("kl", []).append(total_kl / n_batches)
        hist.setdefault("kl_weight", []).append(kl_w)
        hist.setdefault("acc", []).append(epoch_acc)

        val_loss = None
        val_recon = val_region = val_kl = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            vrecon = vkl = vr_num = vr_den = 0.0
            vnb = 0
            with torch.no_grad():
                for batch_dict in val_loader:
                    vmu, vlogvar, vlogits, vregions = [], [], [], []
                    vsession_recons: list[torch.Tensor] = []
                    for session_name, session_batch in batch_dict.items():
                        windows = session_batch["windows"].to(device)
                        recon_target = session_batch["windows_raw"].to(device)
                        regions = session_batch["regions"].to(device)
                        out = model(windows, session_name, return_all=True)
                        vmu.append(out["mu"])
                        vlogvar.append(out["logvar"])
                        vlogits.append(out["region_logits"])
                        vregions.append(regions)
                        vsession_recons.append(poisson_nll(out["rates"], recon_target))
                    mu_v = torch.cat(vmu)
                    lv_v = torch.cat(vlogvar)
                    lo_v = torch.cat(vlogits)
                    re_v = torch.cat(vregions)
                    recon_v = torch.stack(vsession_recons).mean()
                    # Validation KL is the plain mean, without the free-bits floor.
                    kl_v = -0.5 * torch.mean(1 + lv_v - mu_v.pow(2) - lv_v.exp())
                    num, den = _weighted_ce_sums(lo_v, re_v, class_weights)
                    vr_num += num
                    vr_den += den
                    vrecon += recon_v.item()
                    vkl += kl_v.item()
                    vnb += 1
            vnb = max(vnb, 1)
            val_recon = vrecon / vnb
            val_region = _weighted_mean(vr_num, vr_den)
            val_kl = vkl / vnb
            val_loss = (
                RECON_WEIGHT * val_recon + region_weight * val_region + kl_w * val_kl
            )
        hist.setdefault("val_loss", []).append(val_loss)
        hist.setdefault("val_recon", []).append(val_recon)
        hist.setdefault("val_region", []).append(val_region)
        hist.setdefault("val_kl", []).append(val_kl)

        if (epoch + 1) % log_every == 0:
            log_fn(
                f"  [Step1] Epoch {epoch + 1:>4}/{num_epochs}  "
                f"loss={total_loss / n_batches:.4f}  "
                f"recon={total_recon / n_batches:.4f}  "
                f"region={total_region / n_batches:.4f}  "
                f"kl={total_kl / n_batches:.4f} (w={kl_w:.3f})  "
                f"acc={epoch_acc:.1f}%"
                + (f"  val={val_loss:.4f}" if val_loss is not None else "")
            )

        if val_loss is not None and warmed_up:
            if val_loss < best_val - es_delta:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                stall = 0
            else:
                stall += 1
                if stall >= es_patience:
                    log_fn(f"      [Step1] early stop @ epoch {epoch + 1}/{num_epochs}")
                    break

    # Restore the weights that minimized val_loss (best-checkpoint early stopping).
    if best_state is not None:
        model.load_state_dict(best_state)
        log_fn(
            f"      [Step1] restored best-val weights from epoch {best_epoch} "
            f"(val_loss={best_val:.4f})"
        )
    hist["best_epoch"] = best_epoch

    return model, hist


def train_step2[M: nn.Module](
    alignment_decoder: M,
    loader: DataLoader,
    device: torch.device,
    log_fn: Callable[[str], None],
    n_classes: int,
    num_epochs: int,
    lr: float,
    class_counts: dict | None = None,
    class_weight_power: float = 1.0,
    contrastive_weight: float = 2.0,
    region_weight: float = 1.0,
    val_loader: DataLoader | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    log_every: int = 50,
) -> tuple[M, dict]:
    """Train the Transformer alignment decoder on frozen Step 1 latents (Step 2).

    Loss = region_weight * CrossEntropy + contrastive_weight * NT-Xent

    Early stopping and the best-checkpoint restore track the validation CrossEntropy
    alone (``val_region``), not the total loss.

    Args:
        alignment_decoder: AlignmentDecoder; trained in place and returned.
        loader:            DataLoader yielding (sequences, regions, session_ids)
                           from LatentSequenceDataset
        device:            torch.device
        log_fn:            callable(str)
        n_classes:         number of region labels
        num_epochs:        training epochs
        lr:                Adam learning rate (STEP2_LEARNING_RATE=3e-4)
        class_counts:      optional dict {class_idx: count} for class weighting
        class_weight_power: exponent on the inverse-frequency class weights.
        contrastive_weight: weight on NT-Xent loss.
        region_weight:     weight on CrossEntropy loss (default 1.0)
        val_loader:        optional validation DataLoader (same format as loader)
        early_stop_patience: epochs without val_region improvement before stopping
                           (None = ``MAIN_EARLY_STOP_PATIENCE``).
        early_stop_min_delta: minimum improvement to reset patience counter
                           (None = ``MAIN_EARLY_STOP_MIN_DELTA``).
        log_every:         print a progress line every this many epochs (default 50)

    Returns:
        (alignment_decoder, history_dict) where history_dict contains per-epoch lists:
        epoch, loss, region, contrastive, acc, val_loss, val_region, val_contrastive.
        The ``val_*`` entries are None when there is no validation loader. If early
        stopping selected an epoch, the returned decoder holds that epoch's weights.
        Also ``best_epoch``: the 1-based epoch whose weights were restored, or None when
        none were.
    """
    optimizer = torch.optim.Adam(alignment_decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    es_patience, es_delta = _es_defaults(early_stop_patience, early_stop_min_delta)

    class_weights = _inverse_frequency_weights(
        class_counts, n_classes, class_weight_power, device, log_fn
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log_fn("    Training Step 2 (Contrastive + Classification on frozen latents)...")

    hist: dict = {}
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch: int | None = None
    stall = 0

    for epoch in range(num_epochs):
        alignment_decoder.train()
        total_loss = total_region = total_contrastive = 0.0
        correct = total_samples = n_batches = 0

        for sequences, regions, session_ids in loader:
            optimizer.zero_grad()
            sequences = sequences.to(device)
            regions = regions.to(device)

            _, proj, logits = alignment_decoder(sequences)
            region_loss = criterion(logits, regions)
            contrastive_loss = compute_contrastive_loss(
                proj,
                regions.cpu().numpy().tolist(),
                session_ids.numpy().tolist(),
                temperature=TEMPERATURE,
            )
            loss = region_weight * region_loss + contrastive_weight * contrastive_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_region += region_loss.item()
            total_contrastive += contrastive_loss.item()
            _, predicted = torch.max(logits, 1)
            total_samples += regions.size(0)
            correct += int((predicted == regions).sum().item())
            n_batches += 1

        scheduler.step()
        n_batches = max(n_batches, 1)

        epoch_acc = 100.0 * correct / max(total_samples, 1)
        hist.setdefault("epoch", []).append(epoch + 1)
        hist.setdefault("loss", []).append(total_loss / n_batches)
        hist.setdefault("region", []).append(total_region / n_batches)
        hist.setdefault("contrastive", []).append(total_contrastive / n_batches)
        hist.setdefault("acc", []).append(epoch_acc)

        val_loss = val_region = val_contrastive = None
        if val_loader is not None and len(val_loader) > 0:
            alignment_decoder.eval()
            vr_num = vr_den = vc = 0.0
            vnb = 0
            with torch.no_grad():
                for sequences, regions, session_ids in val_loader:
                    sequences = sequences.to(device)
                    regions = regions.to(device)
                    _, proj, logits = alignment_decoder(sequences)
                    num, den = _weighted_ce_sums(logits, regions, class_weights)
                    vr_num += num
                    vr_den += den
                    cl = compute_contrastive_loss(
                        proj,
                        regions.cpu().numpy().tolist(),
                        session_ids.numpy().tolist(),
                        temperature=TEMPERATURE,
                    )
                    vc += cl.item()
                    vnb += 1
            val_region = _weighted_mean(vr_num, vr_den)
            val_contrastive = vc / max(vnb, 1)
            val_loss = region_weight * val_region + contrastive_weight * val_contrastive
        hist.setdefault("val_loss", []).append(val_loss)
        hist.setdefault("val_region", []).append(val_region)
        hist.setdefault("val_contrastive", []).append(val_contrastive)

        if (epoch + 1) % log_every == 0:
            log_fn(
                f"  [Step2] Epoch {epoch + 1:>4}/{num_epochs}  "
                f"loss={total_loss / n_batches:.4f}  "
                f"region={total_region / n_batches:.4f}  "
                f"contrastive={total_contrastive / n_batches:.4f}  "
                f"acc={epoch_acc:.1f}%"
                + (
                    f"  val={val_loss:.4f} (r={val_region:.4f} c={val_contrastive:.4f})"
                    if val_loss is not None
                    else ""
                )
            )

        if val_loss is not None:
            metric = val_region if val_region is not None else val_loss
            if metric < best_val - es_delta:
                best_val = metric
                best_state = copy.deepcopy(alignment_decoder.state_dict())
                best_epoch = epoch + 1
                stall = 0
            else:
                stall += 1
                if stall >= es_patience:
                    log_fn(f"      [Step2] early stop @ epoch {epoch + 1}/{num_epochs}")
                    break

    # Restore the weights that minimized the val metrics
    if best_state is not None:
        alignment_decoder.load_state_dict(best_state)
        log_fn(f"      [Step2] restored best-val weights (val_metric={best_val:.4f})")
    hist["best_epoch"] = best_epoch

    return alignment_decoder, hist
