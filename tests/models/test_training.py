"""Tests for src/models/training.py.

Accepted gaps: whether training improves anything (one-epoch budgets cannot tell); the
progress-log wording.
"""

from __future__ import annotations

import pytest
import torch

from src.models.alignment import AlignmentDecoder
from src.models.config import KL_WEIGHT, RECON_WEIGHT
from src.models.encoder import SharedTemporalClassifier
from src.models.training import (
    _es_defaults,
    _inverse_frequency_weights,
    _step1_param_groups,
    train_step1,
    train_step2,
)

_LD, _DM, _HD, _NC, _BATCH, _T, _SL = 4, 8, 8, 3, 2, 3, 3
_DEVICE = torch.device("cpu")


def _encoder() -> SharedTemporalClassifier:
    """Build a tiny single-session BC-VAE encoder."""
    configs = {
        "S1": {
            "n_total": 4,
            "shared_indices": [0, 1],
            "specific_indices": [2, 3],
            "n_specific": 2,
        }
    }
    return SharedTemporalClassifier(
        2, configs, _NC, shared_dim=_DM, latent_dim=_LD, factor_dim=4, hidden_dim=_HD
    )


def _decoder() -> AlignmentDecoder:
    """Build a tiny Transformer alignment decoder."""
    return AlignmentDecoder(
        latent_dim=_LD,
        n_classes=_NC,
        seq_len=_SL,
        n_heads=2,
        n_layers=1,
        d_model=_DM,
        hidden_dim=_HD,
    )


def _step1_batch():
    """Return one Step-1 batch shaped as hybrid_collate_fn emits it."""
    # Mirrors hybrid_collate_fn: windows_raw is the Poisson reconstruction target.
    return {
        "S1": {
            "windows": torch.zeros(_BATCH, _T, 4),
            "windows_raw": torch.zeros(_BATCH, _T, 4),
            "regions": torch.zeros(_BATCH, dtype=torch.long),
        }
    }


def _step2_batch():
    """Return one Step-2 batch of (sequences, regions, session ids)."""
    return (
        torch.zeros(_BATCH, _SL, _LD),
        torch.zeros(_BATCH, dtype=torch.long),
        torch.zeros(_BATCH, dtype=torch.long),
    )


def test_total_loss_is_the_documented_weighted_sum():
    """Loss = RECON_WEIGHT*recon + region_weight*region + kl_w*kl.

    Recomposing it from the recorded components pins the weighting itself. A
    region_weight silently ignored (the failure that lets the decoder learn the
    majority region and nothing else) shows up here as an arithmetic mismatch.
    """
    torch.manual_seed(0)
    _, hist = train_step1(
        _encoder(),
        [_step1_batch()],  # one batch, so the epoch means are exact
        _DEVICE,
        lambda s: None,
        _NC,
        num_epochs=1,
        lr=1e-3,
        region_weight=2.5,
    )
    expected = (
        RECON_WEIGHT * hist["recon"][0]
        + 2.5 * hist["region"][0]
        + hist["kl_weight"][0] * hist["kl"][0]
    )
    assert hist["loss"][0] == pytest.approx(expected, rel=1e-5)


def _run_step1_with_val(num_epochs, warmup):
    """Train Step 1 with a val loader whose first eligible epoch is always the best."""
    logs: list[str] = []
    torch.manual_seed(0)
    _, hist = train_step1(
        _encoder(),
        [_step1_batch()],
        _DEVICE,
        logs.append,
        _NC,
        num_epochs=num_epochs,
        lr=1e-3,
        val_loader=[_step1_batch()],
        early_stop_patience=1,
        early_stop_min_delta=1e10,  # only the first candidate epoch can "improve"
        kl_warmup_epochs=warmup,
    )
    return hist, logs


def test_step1_early_stopping_skips_the_kl_warmup():
    """No warm-up epoch is selected or spends patience, so the first candidate is the ramp's end."""
    hist, _ = _run_step1_with_val(num_epochs=10, warmup=3)
    assert hist["kl_weight"] == pytest.approx(
        [KL_WEIGHT / 3, 2 * KL_WEIGHT / 3, KL_WEIGHT, KL_WEIGHT]
    )
    assert hist["best_epoch"] == 3
    assert len(hist["loss"]) == 4  # epoch 3 selected, epoch 4 exhausts patience 1


def test_step1_without_a_finished_warmup_restores_nothing():
    """A run shorter than the warm-up keeps its final weights and reports no best epoch."""
    hist, logs = _run_step1_with_val(num_epochs=2, warmup=5)
    assert hist["best_epoch"] is None
    assert not any("restored best-val weights" in m for m in logs)


def test_train_step2_restores_best_val_weights():
    """train_step2 returns the same decoder carrying best-val weights, and logs the restore."""
    # With a val_loader the best-val weights must be restored before return,
    # rather than the final / early-stopped epoch's.
    logs: list[str] = []
    decoder = _decoder()
    out_dec, _ = train_step2(
        decoder,
        [_step2_batch()],
        _DEVICE,
        logs.append,
        _NC,
        num_epochs=10,
        lr=3e-4,
        val_loader=[_step2_batch()],
        early_stop_patience=1,
        early_stop_min_delta=1e10,
    )
    assert out_dec is decoder
    assert any("restored best-val weights" in m for m in logs)


def test_rare_class_gets_proportionally_more_weight():
    """Full inverse-frequency weighting is what stops the decoder
    from simply learning the majority region."""
    w = _inverse_frequency_weights({0: 900, 1: 100}, 2, power=1.0, device=_DEVICE)
    assert w[1] / w[0] == pytest.approx(9.0)
    assert w.sum().item() == pytest.approx(2.0)  # renormalized to n_classes


class TestInverseFrequencyWeights:
    def test_no_counts_means_uniform_weights(self):
        """Without class counts every class weighs 1."""
        w = _inverse_frequency_weights(None, 3, power=1.0, device=_DEVICE)
        assert w.tolist() == [1.0, 1.0, 1.0]

    def test_power_tempers_the_ratio(self):
        """Power 0.5 takes the square root of the inverse-frequency ratio: sqrt(9) = 3."""
        w = _inverse_frequency_weights({0: 900, 1: 100}, 2, power=0.5, device=_DEVICE)
        assert w[1] / w[0] == pytest.approx(3.0)
        assert w.sum().item() == pytest.approx(2.0)

    def test_absent_and_out_of_range_classes(self):
        """A class never seen gets weight 0; a label beyond n_classes is ignored, though its
        count still enters the total. Raw weights: total 400 / (3 * 100) and 400 / (3 * 200)."""
        logs: list[str] = []
        w = _inverse_frequency_weights(
            {0: 100, 2: 200, 5: 100}, 3, power=1.0, device=_DEVICE, log_fn=logs.append
        )
        raw = torch.tensor([400 / 300, 0.0, 400 / 600])
        torch.testing.assert_close(w, raw / raw.sum() * 3)
        assert len(logs) == 1 and "Class weights" in logs[0]

    def test_a_zero_count_does_not_divide_by_zero(self):
        """A listed class with zero windows takes raw weight 1 rather than infinity."""
        w = _inverse_frequency_weights({0: 0, 1: 4}, 2, power=1.0, device=_DEVICE)
        raw = torch.tensor([1.0, 4 / (2 * 4)])
        torch.testing.assert_close(w, raw / raw.sum() * 2)


def test_param_groups_regularize_the_classifier_separately():
    """The classifier head gets its own decay; other matrices decay; biases and norms do not;
    every trainable parameter lands in exactly one group."""
    model = _encoder()
    model.fc_g0.weight.requires_grad_(False)
    groups = _step1_param_groups(model, weight_decay=0.1, classifier_weight_decay=0.7)
    ids = [{id(p) for p in g["params"]} for g in groups]
    decay, no_decay, clf = ids
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0, 0.7]
    assert clf == {id(p) for p in model.classifier.parameters()}
    assert (
        id(model.encoder.fc_mu.weight) in decay
        and id(model.encoder.fc_mu.bias) in no_decay
    )
    assert id(model.fc_g0.weight) not in decay | no_decay | clf  # frozen
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert decay | no_decay | clf == trainable
    assert sum(map(len, ids)) == len(trainable)


def test_param_groups_without_a_classifier_drop_the_empty_group():
    """A model with no classifier head has no (empty) classifier group."""
    groups = _step1_param_groups(torch.nn.Linear(2, 2), 0.1, 0.7)
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]


def test_early_stop_defaults_apply_only_when_unset():
    """None falls back to the configured defaults; an explicit 0 is kept."""
    from src.models.config import MAIN_EARLY_STOP_MIN_DELTA, MAIN_EARLY_STOP_PATIENCE

    assert _es_defaults(None, None) == (
        MAIN_EARLY_STOP_PATIENCE,
        MAIN_EARLY_STOP_MIN_DELTA,
    )
    assert _es_defaults(0, 0.0) == (0, 0.0)


def _step1(num_epochs, **kw):
    """Train Step 1 on the fixed one-batch loader from a fixed seed."""
    torch.manual_seed(0)
    model, hist = train_step1(
        _encoder(), [_step1_batch()], _DEVICE, kw.pop("log_fn", lambda s: None), _NC,
        num_epochs=num_epochs, lr=1e-2, **kw,
    )  # fmt: skip
    return model, hist


def test_free_bits_floor_every_latent_dimension():
    """With a floor far above any real KL, every dimension is clamped to it: kl == floor."""
    _, hist = _step1(1, kl_free_bits=50.0)
    assert hist["kl"][0] == pytest.approx(50.0)


def test_step1_restores_exactly_the_best_epochs_weights():
    """A run whose best epoch is 1 returns the weights a one-epoch run ends with: the first
    epoch is identical in both (the cosine schedule only changes the rate after epoch 1)."""
    kw = dict(
        val_loader=[_step1_batch()], early_stop_patience=100, early_stop_min_delta=1e10,
        kl_warmup_epochs=0,
    )  # fmt: skip
    long, hist = _step1(3, **kw)
    short, _ = _step1(1, **kw)
    assert hist["best_epoch"] == 1 and len(hist["loss"]) == 3
    for k, v in short.state_dict().items():
        torch.testing.assert_close(long.state_dict()[k], v, msg=k)


def test_step1_without_validation_keeps_the_last_weights():
    """No validation loader: no checkpoint, no early stop, val columns are None."""
    _, hist = _step1(2)
    assert hist["best_epoch"] is None and hist["val_loss"] == [None, None]
    assert hist["epoch"] == [1, 2]


def test_step1_logs_progress_every_log_every_epochs():
    """A progress line is emitted at epochs that are multiples of log_every."""
    logs: list[str] = []
    _step1(4, log_fn=logs.append, log_every=2)
    assert sum("[Step1] Epoch" in m for m in logs) == 2


def test_step2_total_loss_is_the_documented_weighted_sum():
    """Loss = region_weight * region + contrastive_weight * contrastive, per batch."""
    torch.manual_seed(0)
    _, hist = train_step2(
        _decoder(), [_step2_batch()], _DEVICE, lambda s: None, _NC, num_epochs=1, lr=1e-3,
        contrastive_weight=0.7, region_weight=1.9,
    )  # fmt: skip
    assert hist["loss"][0] == pytest.approx(
        1.9 * hist["region"][0] + 0.7 * hist["contrastive"][0], rel=1e-5
    )


def test_step2_restores_the_best_val_epoch_and_stops_on_patience():
    """With only epoch 1 able to improve and patience 2, training stops after epoch 3 and
    returns the weights of a one-epoch run."""

    def run(epochs, patience):
        """Train Step 2 from a fixed seed with a validation loader."""
        torch.manual_seed(0)
        return train_step2(
            _decoder(), [_step2_batch()], _DEVICE, lambda s: None, _NC, num_epochs=epochs,
            lr=1e-2, val_loader=[_step2_batch()], early_stop_patience=patience,
            early_stop_min_delta=1e10, log_every=1,
        )  # fmt: skip

    long, hist = run(10, 2)
    short, _ = run(1, 2)
    assert len(hist["loss"]) == 3
    for k, v in short.state_dict().items():
        torch.testing.assert_close(long.state_dict()[k], v, msg=k)


def test_step2_without_validation_records_none():
    """No validation loader: validation columns are None and nothing is restored."""
    logs: list[str] = []
    _, hist = train_step2(
        _decoder(), [_step2_batch()], _DEVICE, logs.append, _NC, num_epochs=2, lr=1e-3,
        log_every=1,
    )  # fmt: skip
    assert hist["val_loss"] == [None, None]
    assert not any("restored" in m for m in logs)
    assert sum("[Step2] Epoch" in m for m in logs) == 2
