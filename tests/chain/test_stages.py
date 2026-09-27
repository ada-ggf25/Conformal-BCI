"""Tests for src/chain/stages.py.

One small session is fitted once with a one-epoch budget (shared across the tests); the numbers
the model learns mean nothing, so the tests check how its outputs are wired together: which rows
the sigma inputs come from, the score formula, and that a checkpoint rebuilds the same model.

Accepted gaps: the quality of what the stages learn (real-data validation covers it).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
import torch

from src.chain.stages import ChainModel, TrainBudget
from src.conformal.learned import scale_of
from src.conformal.signals import SIGNALS, standardize_signals
from src.data.config import WINDOW_SIZE
from src.utils.splits import chronological_task_split

N_CLASSES, N_NEURONS, N_WIN = 3, 5, 80
BUDGET = TrainBudget(epochs=1, patience=1, mlp_epochs=2, mlp_patience=1)
DEVICE = torch.device("cpu")


def _config(n=N_NEURONS):
    """A session whose neurons are all session-specific (as in the pooled runs)."""
    return {
        "n_total": n,
        "shared_indices": [],
        "specific_indices": list(range(n)),
        "n_specific": n,
    }


def _session(seed=0):
    """80 windows of Poisson counts tuned to 3 cycling regions, split 60/30/10."""
    rng = np.random.default_rng(seed)
    regions = (np.arange(N_WIN) // 4) % N_CLASSES
    tuning = rng.uniform(0.5, 4.0, size=(N_CLASSES, N_NEURONS))
    counts = rng.poisson(tuning[regions][:, None, :] * np.ones((1, WINDOW_SIZE, 1)))
    tr, va, te = chronological_task_split(np.ones(N_WIN, bool), 0.6, 0.3, 0.1)
    return {
        "windows": counts.astype("float32"),
        "regions": regions,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
    }


@cache
def _fitted():
    """A model fitted once on session S1, with the inputs fit returned and the session."""
    torch.manual_seed(0)
    model = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    d = _session()
    inputs = model.fit("S1", d, from_scratch=True, budget=BUDGET, alpha=0.1, seed=0)
    return model, inputs, d


def test_sigma_inputs_are_the_train_and_calibration_rows():
    """fit returns signals, probabilities and labels of exactly the train and val windows,
    under the fitted model, and freezes the standardization on the train signals."""
    model, inputs, d = _fitted()
    fw = model.forward("S1", d)
    for key, idx in (("tr", d["train_idx"]), ("va", d["val_idx"])):
        np.testing.assert_array_equal(inputs[key]["labels"], d["regions"][idx])
        np.testing.assert_allclose(inputs[key]["probs"], fw["probs"][idx], rtol=1e-6)
        assert inputs[key]["u"].shape == (len(idx), len(SIGNALS))
    np.testing.assert_allclose(model.stats.mean, inputs["tr"]["u"].mean(0))
    assert model.sigma is not None


def test_forward_probabilities_are_distributions():
    """Softmax outputs are non-negative and sum to one (to float32 precision) per window."""
    model, _, d = _fitted()
    p = model.forward("S1", d)["probs"]
    assert p.shape == (N_WIN, N_CLASSES) and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-6)  # softmax runs in float32


def test_block_selects_the_requested_rows():
    """A block holds the probabilities and labels of the given windows, in the given order."""
    model, _, d = _fitted()
    idx = np.array([5, 2, 70])
    blk = model.block("S1", d, idx)
    np.testing.assert_allclose(blk["probs"], model.forward("S1", d)["probs"][idx])
    assert blk["labels"].tolist() == d["regions"][idx].tolist()
    assert blk["u"].shape == (3, len(SIGNALS))


def test_score_is_one_minus_true_probability_over_sigma():
    """The deployed score divides 1 - p_y by the net's sigma on the standardized signals."""
    model, _, d = _fitted()
    blk = model.block("S1", d, d["test_idx"])
    out = model.score(blk)
    sigma = scale_of(model.sigma, standardize_signals(blk["u"], model.stats))
    p_true = blk["probs"][np.arange(len(blk["labels"])), blk["labels"]]
    np.testing.assert_allclose(out["sigma"], sigma)
    np.testing.assert_allclose(out["scores"], (1.0 - p_true) / sigma)


def test_score_without_a_sigma_is_standard_conformal():
    """A model with no trained sigma (the fresh baseline) scores with sigma = 1."""
    torch.manual_seed(0)
    fresh = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    blk = {
        "probs": np.array([[0.2, 0.5, 0.3]]),
        "u": np.zeros((1, 4)),
        "labels": np.array([1]),
    }
    out = fresh.score(blk)
    assert out["sigma"].tolist() == [1.0] and out["scores"].tolist() == [0.5]


def test_score_with_an_explicit_branch_uses_that_net():
    """A sweep branch passed to score replaces the deployed sigma for that call only."""
    model, inputs, d = _fitted()
    branch = model.train_sigma(inputs, 0.5, None, BUDGET, seed=1)
    blk = model.block("S1", d, d["test_idx"])
    np.testing.assert_allclose(
        model.score(blk, branch)["sigma"],
        scale_of(branch, standardize_signals(blk["u"], model.stats)),
    )


def test_fit_without_sigma_training_keeps_sigma_unset():
    """The fresh baseline is fitted with train_sigma=False: no sigma, but frozen stats."""
    torch.manual_seed(0)
    m = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    m.fit("S1", _session(), from_scratch=True, budget=BUDGET, alpha=0.1, seed=0,
          train_sigma=False)  # fmt: skip
    assert m.sigma is None and m.stats is not None


def test_state_round_trip_rebuilds_the_same_model():
    """A model restored from state() into a new ChainModel, including a session added after
    construction, gives identical probabilities and scores."""
    model, _, d = _fitted()
    model.add_session("S2", _config(4))
    state = model.state()
    assert state["sessions"] == ["S1", "S2"]
    torch.manual_seed(123)  # a different initialization, fully overwritten by load
    other = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    other.load(state, {"S1": _config(), "S2": _config(4)})
    assert other.sessions == ["S1", "S2"]
    blk_a, blk_b = (
        model.block("S1", d, d["test_idx"]),
        other.block("S1", d, d["test_idx"]),
    )
    np.testing.assert_array_equal(blk_a["probs"], blk_b["probs"])
    np.testing.assert_array_equal(
        model.score(blk_a)["scores"], other.score(blk_b)["scores"]
    )
    np.testing.assert_array_equal(other.stats.std, model.stats.std)


def test_state_of_an_untrained_model_has_no_sigma():
    """Before any fit there is no sigma or stats to save, and loading that keeps them unset."""
    m = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    state = m.state()
    assert state["sigma"] is None and state["stats"] is None
    m.load(state, {"S1": _config()})
    assert m.sigma is None and m.stats is None


def test_adding_a_known_session_again_changes_nothing():
    """Re-adding a session (e.g. on resume) must not re-initialize its layers."""
    m = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    m.add_session("S2", _config(4))
    before = {k: v.clone() for k, v in m.step1.state_dict().items()}
    m.add_session("S2", _config(4))
    m.add_session("S1", _config())
    assert m.sessions == ["S1", "S2"]
    for k, v in m.step1.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_train_sigma_needs_frozen_statistics():
    """Sigma cannot be trained before fit has frozen the signal standardization."""
    m = ChainModel("S1", _config(), N_CLASSES, DEVICE)
    with pytest.raises(AssertionError):
        m.train_sigma({"tr": {}, "va": {}}, 0.1, None, BUDGET, 0)
