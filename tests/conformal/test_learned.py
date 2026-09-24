"""Tests for src/conformal/learned.py."""

from __future__ import annotations

import numpy as np

from src.conformal.learned import scale_of, signal_sensitivity, train_scale

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
    """The reported per-signal attribution."""
    z, p, y = _signals(300), _probs(300), _labels(300)
    net, _ = train_scale(z, p, y, alpha=0.1, epochs=30, seed=0, device="cpu")
    raw = _signals(5, seed=99)
    grad = signal_sensitivity(net, raw, np.zeros(N_SIGNALS), np.ones(N_SIGNALS))

    eps, j = 1e-3, 1  # perturb the second signal only
    raw_plus, raw_minus = raw.copy(), raw.copy()
    raw_plus[:, j] += eps
    raw_minus[:, j] -= eps
    fd = (scale_of(net, raw_plus) - scale_of(net, raw_minus)) / (2 * eps)
    np.testing.assert_allclose(grad[:, j], fd, atol=1e-3, rtol=1e-2)
