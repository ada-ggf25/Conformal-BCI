"""Tests for src/conformal/learned.py.

Accepted gaps: whether the trained scale is a *good* one (real-data validation covers it); the
epoch-with-no-batch branch of ``train_scale``, unreachable since the batch size is capped at n.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.conformal.learned import (
    ScaleNet,
    TrainReport,
    _chronological_halves,
    _class_conditional_soft_quantiles,
    _conformal_loss_class_conditional,
    _in_batch_halves,
    scale_of,
    signal_sensitivity,
    signal_sensitivity_for_bundle,
    train_scale,
)

N_CLASSES, N_SIGNALS = 6, 3


def _probs(n, seed=0):
    """Return n rows of sampled class posteriors."""
    return np.random.RandomState(seed).dirichlet(np.ones(N_CLASSES), size=n)


def _labels(n, seed=1):
    """Return n random true-class labels."""
    return np.random.RandomState(seed).randint(0, N_CLASSES, size=n)


def _signals(n, seed=2):
    """Return an (n, N_SIGNALS) matrix of standard-normal uncertainty signals."""
    return np.random.RandomState(seed).randn(n, N_SIGNALS)


def test_early_stopping_restores_best_weights():
    """The returned net must carry the best epoch's weights, not the last one's."""
    z, p, y = _signals(256), _probs(256), _labels(256)
    vz, vp, vy = _signals(128, 5), _probs(128, 5), _labels(128, 5)
    kw = dict(alpha=0.1, n_classes=N_CLASSES, seed=0, device="cpu")
    net, report = train_scale(
        z,
        p,
        y,
        epochs=30,
        val_signals_z=vz,
        val_probs=vp,
        val_labels=vy,
        early_stop_patience=3,
        **kw,
    )
    assert report.best_epoch is not None

    assert report.best_epoch < len(report.earlystop_loss) - 1
    ref, _ = train_scale(
        z,
        p,
        y,
        epochs=report.best_epoch + 1,
        val_signals_z=vz,
        val_probs=vp,
        val_labels=vy,
        early_stop_patience=None,
        **kw,
    )
    np.testing.assert_allclose(scale_of(net, z), scale_of(ref, z), rtol=0, atol=0)


def test_gradient_matches_finite_difference():
    """The reported per-signal attribution is d(log sigma)/d(signal)."""
    z, p, y = _signals(300), _probs(300), _labels(300)
    net, _ = train_scale(z, p, y, alpha=0.1, epochs=30, seed=0, device="cpu")
    raw = _signals(5, seed=99)
    grad = signal_sensitivity(net, raw, np.zeros(N_SIGNALS), np.ones(N_SIGNALS))

    eps, j = 1e-3, 1  # perturb the second signal only
    raw_plus, raw_minus = raw.copy(), raw.copy()
    raw_plus[:, j] += eps
    raw_minus[:, j] -= eps
    fd = (np.log(scale_of(net, raw_plus)) - np.log(scale_of(net, raw_minus))) / (
        2 * eps
    )
    np.testing.assert_allclose(grad[:, j], fd, atol=1e-3, rtol=1e-2)


def test_chronological_halves_put_each_sessions_past_in_cal():
    """Within each group the cal rows all precede the ev rows, and they split it in half."""
    groups = torch.tensor([1, 0, 1, 0, 1, 0, 2, 1])
    times = torch.tensor([7, 3, 2, 9, 5, 1, 4, 0])
    cal, ev = _chronological_halves(groups, times)
    assert sorted(cal.tolist() + ev.tolist()) == list(range(len(groups)))
    for g in groups.unique().tolist():
        c = times[cal][groups[cal] == g]
        e = times[ev][groups[ev] == g]
        m = int((groups == g).sum())
        assert len(c) == m // 2 and len(e) == m - m // 2
        if len(c):
            assert c.max() < e.min()


def test_chronological_split_trains_and_is_deterministic():
    """The chronological mode runs end to end and reproduces under a fixed seed."""
    z, p, y = _signals(256), _probs(256), _labels(256)
    vz, vp, vy = _signals(128, 5), _probs(128, 5), _labels(128, 5)
    g, vg = np.repeat(np.arange(4), 64), np.repeat(np.arange(4), 32)
    kw = dict(
        alpha=0.1,
        epochs=10,
        val_signals_z=vz,
        val_probs=vp,
        val_labels=vy,
        early_stop_patience=3,
        n_classes=N_CLASSES,
        split="chronological",
        groups=g,
        val_groups=vg,
        seed=0,
        device="cpu",
    )
    a, rep = train_scale(z, p, y, **kw)
    b, _ = train_scale(z, p, y, **kw)
    assert rep.epochs >= 1
    np.testing.assert_array_equal(scale_of(a, z), scale_of(b, z))


def test_chronological_split_requires_groups():
    """Asking for the chronological split without session ids is an error."""
    z, p, y = _signals(64), _probs(64), _labels(64)
    with pytest.raises(ValueError, match="groups"):
        train_scale(z, p, y, epochs=1, split="chronological", device="cpu")


def test_init_net_continues_from_a_trained_net_without_mutating_it():
    """Zero epochs from init_net returns init_net's function; the passed net is not trained."""
    z, p, y = _signals(128), _probs(128), _labels(128)
    base, _ = train_scale(z, p, y, epochs=5, seed=0, device="cpu")
    before = scale_of(base, z)
    same, _ = train_scale(z, p, y, epochs=0, init_net=base, seed=0, device="cpu")
    np.testing.assert_array_equal(scale_of(same, z), before)
    moved, _ = train_scale(z, p, y, epochs=5, init_net=base, seed=1, device="cpu")
    np.testing.assert_array_equal(scale_of(base, z), before)
    assert not np.array_equal(scale_of(moved, z), before)


# --------------------------------------------------------------------------- ScaleNet


def test_a_fresh_scale_net_is_exactly_standard_conformal():
    """The zero-initialized head gives sigma = 1 for every window before any training."""
    net = ScaleNet(N_SIGNALS)
    assert np.all(scale_of(net, _signals(20) * 100) == 1.0)


def test_scale_is_clamped_to_a_bounded_positive_range():
    """A diverging head cannot drive sigma to 0 or inf: log sigma is clamped to [-3, 3].

    With every weight and bias set to +-1, f(x) = +-(4 * relu(x + 1) + 1): 1 (or -1) far
    below zero, and far beyond the clamp at x = 0 (5) and above.
    """
    for sign in (1.0, -1.0):
        net = ScaleNet(1, hidden=4, depth=1)
        with torch.no_grad():
            for p in net.net[0].parameters():
                p.fill_(1.0)
            for p in net.net[-1].parameters():
                p.fill_(sign)
        s = scale_of(net, np.array([[-1e6], [0.0], [1e6]]))
        np.testing.assert_allclose(
            np.log(s), [sign * 1.0, sign * 3.0, sign * 3.0], rtol=1e-6
        )


def test_depth_zero_is_a_linear_scale():
    """With no hidden layer log sigma is a linear function of the signals."""
    net = ScaleNet(2, depth=0)
    with torch.no_grad():
        net.net[0].weight.copy_(torch.tensor([[0.5, -1.0]]))
        net.net[0].bias.fill_(0.25)
    s = scale_of(net, np.array([[1.0, 1.0], [2.0, 0.0]]))
    np.testing.assert_allclose(np.log(s), [-0.25, 1.25], rtol=1e-6)


def test_scale_net_needs_at_least_one_signal():
    """A scale with no inputs is refused."""
    with pytest.raises(ValueError, match="n_signals"):
        ScaleNet(0)


def test_collapse_flag_reads_the_scale_spread():
    """A net that never moved from sigma = 1 has no spread and is flagged as collapsed."""
    assert TrainReport(0, [], [], [], scale_std=0.0).collapsed_to_standard
    assert not TrainReport(0, [], [], [], scale_std=0.1).collapsed_to_standard


# --------------------------------------------------------------------------- the loss


def test_soft_quantile_per_class_with_a_sentinel_for_absent_classes():
    """Each class's quantile comes from its own cal rows; a class with no cal row gets a huge
    sentinel (freely admitted). torch.quantile(0.9) of (1, 2, 3) interpolates to 2.8."""
    s = torch.tensor([1.0, 2.0, 3.0, 9.0, 0.5])
    labels = torch.tensor([0, 0, 0, 1, 1])
    q = _class_conditional_soft_quantiles(s, labels, torch.tensor([0, 1, 2, 4]), 3, 0.1)
    assert q[0].item() == pytest.approx(2.8)
    assert q[1].item() == pytest.approx(0.5)  # row 3 is not in cal
    assert q[2].item() == 1e6


def test_class_conditional_loss_by_hand():
    """sigma = 1, tau -> 0 makes membership hard. Cal rows give q = (0.2, 0.3). Ev row 2
    (label 0, scores 0.1 / 0.9) has set {0}: covered, size 1. Ev row 3 (label 1, scores
    0.5 / 0.5) has an empty set: not covered. Size mean 0.5; class 1 falls 0.9 short, class
    0 not at all, averaged over the 2 present classes: loss = 0.5 + 20 * 0.9 / 2 = 9.5."""
    probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]])
    labels = torch.tensor([0, 1, 0, 1])
    loss, size, cover = _conformal_loss_class_conditional(
        ScaleNet(1), torch.zeros(4, 1), probs, labels, 2, 0.1, tau=1e-4, kappa=20.0,
        cal_i=torch.tensor([0, 1]), ev_i=torch.tensor([2, 3]),
    )  # fmt: skip
    assert loss.item() == pytest.approx(9.5, abs=1e-4)
    assert size == pytest.approx(0.5, abs=1e-4) and cover == pytest.approx(
        0.5, abs=1e-4
    )


def test_size_weight_scales_only_the_sharpness_term():
    """Halving the size weight lowers the loss by half the mean soft set size."""
    probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.5, 0.5]])
    labels = torch.tensor([0, 1, 0, 1])
    args = (ScaleNet(1), torch.zeros(4, 1), probs, labels, 2, 0.1)
    kw = dict(
        tau=1e-4, kappa=20.0, cal_i=torch.tensor([0, 1]), ev_i=torch.tensor([2, 3])
    )
    full, size, _ = _conformal_loss_class_conditional(*args, **kw)
    half, _, _ = _conformal_loss_class_conditional(*args, **kw, size_weight=0.5)
    assert (full - half).item() == pytest.approx(0.5 * size, abs=1e-5)


def test_random_halves_partition_the_batch():
    """The random split puts floor(n/2) rows in cal and the rest in ev, each row once."""
    cal, ev = _in_batch_halves(
        9, "random", torch.Generator().manual_seed(0), None, None
    )
    assert len(cal) == 4 and len(ev) == 5
    assert sorted(cal.tolist() + ev.tolist()) == list(range(9))


def test_chronological_halves_without_groups_is_an_error():
    """The chronological split cannot run without per-row groups and times."""
    with pytest.raises(ValueError, match="groups and times"):
        _in_batch_halves(4, "chronological", torch.Generator(), None, torch.arange(4))


# --------------------------------------------------------------------------- train_scale


@pytest.mark.parametrize(
    ("n_sig", "n_prob", "n_lab", "kw", "message"),
    [
        (40, 40, 39, {}, "length mismatch"),
        (31, 31, 31, {}, ">= 32"),
        (40, 40, 40, {"split": "bogus"}, "split must be"),
        (40, 40, 40, {"split": "chronological", "groups": np.zeros(39)}, "groups"),
        (40, 40, 40, {"split": "chronological", "groups": np.zeros(40),
                      "val_signals_z": np.zeros((32, N_SIGNALS))}, "val_groups"),
    ],
)  # fmt: skip
def test_train_scale_rejects_bad_inputs(n_sig, n_prob, n_lab, kw, message):
    """Misaligned inputs, too few windows, an unknown split, or missing groups all raise."""
    with pytest.raises(ValueError, match=message):
        train_scale(_signals(n_sig), _probs(n_prob), _labels(n_lab), epochs=1, **kw)


def test_zero_epochs_returns_standard_conformal():
    """With no training the net is untouched: sigma = 1 and the report says collapsed."""
    z, p, y = _signals(64), _probs(64), _labels(64)
    net, report = train_scale(z, p, y, epochs=0)
    assert np.all(scale_of(net, z) == 1.0)
    assert report.collapsed_to_standard and report.epochs == 0
    assert report.earlystop_loss is None and report.best_epoch is None


def test_report_records_one_entry_per_epoch():
    """Every epoch contributes one loss, soft size and soft coverage to the report."""
    z, p, y = _signals(96), _probs(96), _labels(96)
    _, report = train_scale(z, p, y, epochs=4, batch_size=32)
    assert report.epochs == 4
    assert len(report.loss) == len(report.val_size) == len(report.val_coverage) == 4
    assert all(0.0 <= c <= 1.0 for c in report.val_coverage)
    assert all(0.0 <= s <= N_CLASSES for s in report.val_size)


def test_training_is_reproducible_under_a_seed():
    """Two runs with the same seed give the same scale; a different seed does not."""
    z, p, y = _signals(128), _probs(128), _labels(128)
    a, _ = train_scale(z, p, y, epochs=5, seed=3)
    b, _ = train_scale(z, p, y, epochs=5, seed=3)
    c, _ = train_scale(z, p, y, epochs=5, seed=4)
    np.testing.assert_array_equal(scale_of(a, z), scale_of(b, z))
    assert not np.array_equal(scale_of(a, z), scale_of(c, z))


# --------------------------------------------------------------------------- sensitivity


def _net(seed):
    """A small scale net with non-trivial random weights (its head is not zero)."""
    torch.manual_seed(seed)
    net = ScaleNet(N_SIGNALS, hidden=8, depth=1)
    with torch.no_grad():
        net.net[-1].weight.normal_(0.0, 0.5)
    return net


def test_bundle_sensitivity_uses_each_sessions_own_net_and_rows(tmp_path):
    """Session i's net, with its own standardization, is differentiated on the windows with
    win_session_idx == i; absolute gradients are then averaged over all windows."""
    names = ["a", "b", "c"]
    nets = [_net(0), _net(1)]
    stats = [
        (np.zeros(3), np.ones(3)),
        (np.array([1.0, -1.0, 0.5]), np.array([2.0, 0.5, 1.0])),
    ]
    raw = _signals(7, seed=11)
    sess_idx = np.array([0, 1, 0, 1, 1, 0, 1])
    torch.save(
        {
            "signals": names, "hidden": 8, "depth": 1,
            "sessions": [
                {"session": f"S{i}", "state_dict": nets[i].state_dict(),
                 "signal_mean": list(stats[i][0]), "signal_std": list(stats[i][1])}
                for i in range(2)
            ],
        },
        tmp_path / "sigma_nets.pt",
    )  # fmt: skip
    np.savez(
        tmp_path / "arrays.npz",
        win_session_idx=sess_idx,
        **{f"win_{n}": raw[:, j] for j, n in enumerate(names)},
    )
    out = signal_sensitivity_for_bundle(
        {
            "_sigma_nets_path": tmp_path / "sigma_nets.pt",
            "_arrays_path": tmp_path / "arrays.npz",
        }
    )
    grads = np.vstack(
        [signal_sensitivity(nets[i], raw[sess_idx == i], *stats[i]) for i in range(2)]
    )
    assert out["signals"] == names
    np.testing.assert_allclose(
        out["mean_abs_log_grad"], np.abs(grads).mean(0), rtol=1e-6
    )
    swapped = np.vstack(
        [
            signal_sensitivity(nets[1 - i], raw[sess_idx == i], *stats[i])
            for i in range(2)
        ]
    )
    assert not np.allclose(out["mean_abs_log_grad"], np.abs(swapped).mean(0))


def test_bundle_sensitivity_needs_both_sidecar_files():
    """Without the sigma nets or the arrays there is nothing to differentiate."""
    assert (
        signal_sensitivity_for_bundle({"_sigma_nets_path": None, "_arrays_path": "x"})
        is None
    )
    assert signal_sensitivity_for_bundle({}) is None


def test_sensitivity_standardizes_with_the_given_statistics():
    """d log sigma / d z is taken at z = (raw - mean) / std, not at the raw values."""
    net = _net(2)
    raw = _signals(4, seed=3)
    mean, std = np.array([0.5, -0.5, 1.0]), np.array([2.0, 1.0, 0.5])
    np.testing.assert_allclose(
        signal_sensitivity(net, raw, mean, std),
        signal_sensitivity(net, (raw - mean) / std, np.zeros(3), np.ones(3)),
        rtol=1e-6,
    )


def test_chronological_halves_send_a_single_row_group_to_ev():
    """A group with one row has no past to calibrate on: it is scored, never used for q."""
    cal, ev = _chronological_halves(torch.tensor([5, 5, 7]), torch.tensor([0, 1, 0]))
    assert cal.tolist() == [0] and sorted(ev.tolist()) == [1, 2]


def test_soft_quantile_passes_gradient_to_the_scale():
    """The quantile is differentiable in the scores, so the shared scale gets a gradient."""
    scale = torch.tensor(2.0, requires_grad=True)
    s = torch.tensor([1.0, 2.0, 3.0]) / scale
    q = _class_conditional_soft_quantiles(s, torch.zeros(3, dtype=torch.long),
                                          torch.arange(3), 1, 0.1)  # fmt: skip
    q.sum().backward()
    assert scale.grad is not None and scale.grad.item() < 0  # larger scale, smaller q


def test_early_stopping_with_no_epochs_keeps_standard_conformal():
    """With a held-out block but zero epochs there is no checkpoint to restore: sigma = 1."""
    z, p, y = _signals(64), _probs(64), _labels(64)
    net, report = train_scale(
        z, p, y, epochs=0, val_signals_z=z, val_probs=p, val_labels=y
    )
    assert np.all(scale_of(net, z) == 1.0) and report.best_epoch is None
