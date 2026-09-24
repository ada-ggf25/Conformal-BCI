"""Training loops for Step 1 (BC-VAE encoder) and Step 2 (Transformer alignment decoder).

Adapted from the Liu Lab's cross-session alignment codebase.
"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    KL_WEIGHT,
    MAIN_EARLY_STOP_MIN_DELTA,
    MAIN_EARLY_STOP_PATIENCE,
    RECON_WEIGHT,
    REGION_WEIGHT,
    TEMPERATURE,
)
from .losses import compute_contrastive_loss, poisson_nll


def _es_defaults(patience, min_delta):
    p = MAIN_EARLY_STOP_PATIENCE if patience is None else patience
    d = MAIN_EARLY_STOP_MIN_DELTA if min_delta is None else min_delta
    return p, d


def _step1_param_groups(
    model: nn.Module, weight_decay: float, classifier_weight_decay: float
) -> list[dict]:
    """Adam param groups that regularize the classifier head separately."""
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
    """Inverse-frequency CrossEntropy weights: w_c = (total / (K * count_c)) ** power."""
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
    log_every: int = 50,
) -> tuple[M, dict]:
    """Train the BC-VAE encoder (Step 1).

    Loss = RECON_WEIGHT * Poisson_NLL + region_weight * CrossEntropy + kl_w * KL,
    where ``kl_w`` ramps ``kl_weight`` linearly over the first 10 epochs.

    Args:
        model:          SharedTemporalClassifier
        loader:         DataLoader yielding batch_dict (from HybridClassificationDataset)
        device:         torch.device
        log_fn:         callable(str) for progress output
        n_classes:      number of region labels
        num_epochs:     training epochs
        lr:             Adam learning rate (STEP1_LEARNING_RATE=1e-3)
        class_counts:   optional dict {class_idx: count} for inverse-frequency weighting
        val_loader:     optional DataLoader for validation loss
        early_stop_patience: epochs without val_loss improvement before stopping
        early_stop_min_delta: minimum improvement to reset patience counter
        weight_decay:   AdamW L2 decay on the encoder/generator (0.0 = none)
        classifier_weight_decay: AdamW L2 decay applied to the region-classifier head only
        region_weight:  weight on the region CrossEntropy term.
        class_weight_power: exponent applied to inverse-frequency class weights.
        kl_weight:      weight on the KL term.
        kl_free_bits:   per-dimension KL floor in nats (0.0 = off).
        log_every:      print a progress line every this many epochs (default 50)

    Returns:
        (model, history_dict) where history_dict contains per-epoch lists:
        epoch, loss, recon, region, kl, kl_weight, acc, val_loss, val_recon,
        val_region and val_kl.
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
    stall = 0

    for epoch in range(num_epochs):
        kl_w = kl_weight * min(1.0, (epoch + 1) / 10.0)  # linear warm-up over 10 epochs
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
            vtot = vrecon = vregion = vkl = 0.0
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
                    kl_v = -0.5 * torch.mean(1 + lv_v - mu_v.pow(2) - lv_v.exp())
                    region_v = criterion(lo_v, re_v)
                    vtot += (
                        RECON_WEIGHT * recon_v + region_weight * region_v + kl_w * kl_v
                    ).item()
                    vrecon += recon_v.item()
                    vregion += region_v.item()
                    vkl += kl_v.item()
                    vnb += 1
            vnb = max(vnb, 1)
            val_loss = vtot / vnb
            val_recon = vrecon / vnb
            val_region = vregion / vnb
            val_kl = vkl / vnb
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

        if val_loss is not None:
            if val_loss < best_val - es_delta:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                stall = 0
            else:
                stall += 1
                if stall >= es_patience:
                    log_fn(f"      [Step1] early stop @ epoch {epoch + 1}/{num_epochs}")
                    break

    # Restore the weights that minimized val_loss (best-checkpoint early stopping).
    if best_state is not None:
        model.load_state_dict(best_state)
        log_fn(f"      [Step1] restored best-val weights (val_loss={best_val:.4f})")

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

    Args:
        alignment_decoder: AlignmentDecoder
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
        early_stop_min_delta: minimum improvement to reset patience counter
        log_every:         print a progress line every this many epochs (default 50)

    Returns:
        (alignment_decoder, history_dict) where history_dict contains per-epoch lists:
        epoch, loss, region, contrastive, acc, val_loss, val_region, val_contrastive
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
            vtot = vr = vc = 0.0
            vnb = 0
            with torch.no_grad():
                for sequences, regions, session_ids in val_loader:
                    sequences = sequences.to(device)
                    regions = regions.to(device)
                    _, proj, logits = alignment_decoder(sequences)
                    rl = criterion(logits, regions)
                    cl = compute_contrastive_loss(
                        proj,
                        regions.cpu().numpy().tolist(),
                        session_ids.numpy().tolist(),
                        temperature=TEMPERATURE,
                    )
                    vtot += (region_weight * rl + contrastive_weight * cl).item()
                    vr += rl.item()
                    vc += cl.item()
                    vnb += 1
            val_loss = vtot / max(vnb, 1)
            val_region = vr / max(vnb, 1)
            val_contrastive = vc / max(vnb, 1)
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

    return alignment_decoder, hist
