"""Tests for src/utils/artifacts.py.

Every expected number is pooled or averaged by hand from the small per-window arrays below. The
read-back side of ``build_chain_bundle`` is covered in ``test_bundle_readers.py``.

Accepted gaps: none.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.conformal.signals import SIGNALS
from src.utils.artifacts import (
    LEVEL_CODES,
    build_chain_bundle,
    jsonify,
    method_summary,
    paradigm_slug,
    sigma_nets_payload,
)


class TestParadigmSlug:
    @pytest.mark.parametrize(
        ("paradigm", "slug"),
        [
            ("Nested_Maze+2", "Nested_Maze_2"),
            ("Y_Maze", "Y_Maze"),
            ("  More Water!! ", "More_Water"),
            ("a+-+b", "a_b"),
        ],
    )
    def test_non_alphanumeric_runs_collapse_to_one_underscore(self, paradigm, slug):
        """A slug keeps letters and digits, joins the rest with single underscores."""
        assert paradigm_slug(paradigm) == slug

    @pytest.mark.parametrize("paradigm", ["", "+++", "__"])
    def test_a_paradigm_with_no_name_is_refused(self, paradigm):
        """An empty slug would write the bundle into the parent directory."""
        with pytest.raises(ValueError, match="empty directory name"):
            paradigm_slug(paradigm)


def test_jsonify_makes_numpy_json_safe():
    """Numpy scalars and arrays become Python values, non-finite floats become null, and keys
    become strings, so json.dump accepts the result unchanged."""
    obj = {
        1: np.float32(0.5),
        "arr": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "flags": (np.bool_(True), False),
        "bad": [np.inf, -np.inf, np.nan, np.float64(np.nan)],
        "nested": {"x": np.int16(7), "s": "text", "none": None},
    }
    out = jsonify(obj)
    assert out == {
        "1": 0.5,
        "arr": [[1, 2], [3, 4]],
        "flags": [True, False],
        "bad": [None, None, None, None],
        "nested": {"x": 7, "s": "text", "none": None},
    }
    assert type(out["nested"]["x"]) is int and type(out["flags"][0]) is bool
    json.dumps(out, allow_nan=False)


def _ev(covered, size, error, level):
    """One method's per-window results."""
    return {
        "covered": np.array(covered, float),
        "size_over_k": np.array(size, float),
        "error": np.array(error, float),
        "level": np.array(level),
    }


def test_method_summary_pools_windows_and_averages_sessions():
    """Coverage, size and accuracy pool the 5 windows; the gap and the count below target are
    per session (0.5 and 1.0 against 0.9 -> gaps 0.4 and 0.1, one below)."""
    rows = [
        {"methods": {"m": _ev([1, 0], [0.5, 1.0], [0, 1], ["own", "global"])}},
        {
            "methods": {
                "m": _ev([1, 1, 1], [0.5] * 3, [0, 0, 1], ["own", "borrowed", "own"])
            }
        },
    ]
    s = method_summary(rows, "m", 0.9)
    assert s["n_sessions"] == 2 and s["n_windows"] == 5
    assert s["coverage"] == pytest.approx(4 / 5)
    assert s["size_over_k"] == pytest.approx(3.0 / 5)
    assert s["accuracy"] == pytest.approx(1 - 2 / 5)
    assert s["mean_abs_gap"] == pytest.approx(0.25)
    assert s["sessions_below"] == 1
    assert (s["share_own"], s["share_borrowed"], s["share_global"]) == pytest.approx(
        (3 / 5, 1 / 5, 1 / 5)
    )


def _row(name, first, labels, covered, size, error, level, sweep_cov=None):
    """One tested session's run_chain row, with the same result for both methods."""
    n = len(labels)
    ev = _ev(covered, size, error, level)
    return {
        "session": name,
        "mouse": name.split("_")[0],
        "first_of_mouse": first,
        "n_neurons": 7,
        "labels": np.array(labels),
        "sigma_rel": np.linspace(0.5, 1.5, n),
        "signals": np.arange(n * len(SIGNALS), dtype=float).reshape(n, len(SIGNALS)),
        "methods": {"chain_B3": ev, "fresh": ev},
        "sweep": {
            0.5: {
                "covered": np.array(sweep_cov or covered, float),
                "size_over_k": ev["size_over_k"],
            },
            0.9: {"covered": ev["covered"], "size_over_k": ev["size_over_k"]},
        },
    }


def _bundle(rows, chain=("W_1", "A_2", "B_3"), n_classes=3):
    """build_chain_bundle over one warm-up session and the given rows."""
    data = {
        "W_1": {"regions": np.array([0, 0, 1, 2]), "train_idx": np.array([0, 1]),
                "val_idx": np.array([2]), "test_idx": np.array([], int)},
        "A_2": {"regions": np.array([2, 2, 1]), "train_idx": np.array([0]),
                "val_idx": np.array([1]), "test_idx": np.array([2])},
        "B_3": {"regions": np.array([0, 9, 1]), "train_idx": np.array([0]),
                "val_idx": np.array([], int), "test_idx": np.array([2])},
    }  # fmt: skip
    return build_chain_bundle(
        rows, chain=list(chain), n_warmup=1, session_data=data, n_classes=n_classes,
        paradigm="Y_Maze", seed=7, alpha=0.1, rule="B3", config={"k": 1},
    )  # fmt: skip


class TestBuildChainBundle:
    ROWS = [
        _row("A_2", True, [0, 0, 1], [1, 1, 0], [1 / 3, 2 / 3, 1 / 3], [0, 1, 1],
             ["own", "own", "global"], sweep_cov=[0, 0, 0]),
        _row("B_3", False, [0, 0, 0, 0, 2, 0], [1] * 6, [1 / 3] * 6, [0] * 6,
             ["own"] * 5 + ["borrowed"]),
    ]  # fmt: skip

    def test_class_counts_cover_every_task_window_of_the_whole_chain(self):
        """Region shares come from every chain session's train, val and test windows (the
        warm-up included), never from windows outside the task split (B_3's region 9)."""
        m, _ = _bundle(self.ROWS)
        # W_1: 0,0,1 ; A_2: 2,2,1 ; B_3: 0,1 -> {0: 3, 1: 3, 2: 2}
        assert m["class_counts"] == {0: 3, 1: 3, 2: 2}
        assert m["chain"] == {"warmup": ["W_1"], "config": {"k": 1}}
        assert m["tested_sessions"] == ["A_2", "B_3"]

    def test_decoder_headline_is_pooled_over_test_windows(self):
        """Accuracy 7/9 correct; majority baseline 7 of 9 test windows are region 0."""
        m, _ = _bundle(self.ROWS)
        assert m["decoder"]["raw_accuracy"] == pytest.approx(7 / 9)
        assert m["decoder"]["majority_baseline"] == pytest.approx(7 / 9)
        assert m["decoder"]["chance"] == pytest.approx(1 / 3)

    def test_set_size_is_reported_in_classes(self):
        """The arm's set size is the pooled |C|/K times K: (1+2+1+6*1)/9 classes."""
        m, _ = _bundle(self.ROWS)
        arm = m["conformal"]["arms"]["learned-scale class-conditional"]
        assert arm["set_size"] == pytest.approx(10 / 9)

    def test_per_region_coverage_needs_five_windows(self):
        """Region 0 has 7 test windows (all covered; A_2's miss is its region-1 window), region 1
        one, region 2 one: only region 0 is reported, but every region's count is."""
        pr = _bundle(self.ROWS)[0]["conformal"]["per_region_coverage"]
        assert pr["learned"] == {0: 1.0, 1: None, 2: None}
        assert pr["n_test"] == {0: 7, 1: 1, 2: 1}
        assert pr["size"][0] == pytest.approx((1 + 2 + 5 * 1) / 7)

    def test_sweep_marks_only_the_deployed_target(self):
        """The 90% point is the deployed one; each point pools its own covered arrays."""
        sweep = _bundle(self.ROWS)[0]["conformal"]["retrain_sweep"]["learned"]
        assert [(p["target"], p["deployed"]) for p in sweep] == [
            (0.5, False),
            (0.9, True),
        ]
        assert sweep[0]["coverage"] == pytest.approx(6 / 9)
        assert sweep[1]["coverage"] == pytest.approx(8 / 9)

    def test_session_groups_split_first_and_returning_mice(self):
        """A_2 is its mouse's first session, B_3 a returning one; each group summarizes only
        its own sessions, and a group with no sessions is left out."""
        methods = _bundle(self.ROWS)[0]["conformal"]["methods"]
        assert methods["first"]["chain_B3"]["coverage"] == pytest.approx(2 / 3)
        assert methods["returning"]["chain_B3"]["coverage"] == pytest.approx(1.0)
        assert methods["all"]["fresh"]["n_windows"] == 9
        only_first = _bundle(self.ROWS[:1], chain=("W_1", "A_2"))[0]["conformal"][
            "methods"
        ]
        assert set(only_first) == {"all", "first"}

    def test_per_session_rows(self):
        """Each tested session gets its own coverage, target check and global share."""
        per = _bundle(self.ROWS)[0]["conformal"]["test_coverage"]["per_session"]
        assert [r["session"] for r in per] == ["A_2", "B_3"]
        assert per[0]["coverage"] == pytest.approx(2 / 3) and not per[0]["meets_target"]
        assert per[0]["share_global"] == pytest.approx(1 / 3)
        assert per[1]["meets_target"] and per[1]["n"] == 6
        tc = _bundle(self.ROWS)[0]["conformal"]["test_coverage"]
        assert tc["n_sessions_below_target"] == 1
        assert tc["whole_test"]["meets_target"] is False  # 8/9 < 0.9

    def test_arrays_hold_one_row_per_test_window_in_chain_order(self):
        """Window arrays concatenate sessions in order; levels use the stored integer codes and
        each signal column lands under its own name."""
        _, a = _bundle(self.ROWS)
        assert a["win_session_idx"].tolist() == [0] * 3 + [1] * 6
        assert a["win_region"].tolist() == [0, 0, 1, 0, 0, 0, 0, 2, 0]
        own, borrowed, glob = (LEVEL_CODES[k] for k in ("own", "borrowed", "global"))
        assert a["win_level"].tolist() == [own, own, glob] + [own] * 5 + [borrowed]
        assert a["win_covered_fresh"].tolist() == [1, 1, 0] + [1] * 6
        for j, name in enumerate(SIGNALS):
            np.testing.assert_array_equal(
                a[f"win_{name}"],
                np.r_[self.ROWS[0]["signals"][:, j], self.ROWS[1]["signals"][:, j]],
            )


def test_meeting_the_target_is_inclusive():
    """Coverage exactly at 1 - alpha meets the target: 9 of 10 windows at alpha 0.1."""
    row = _row(
        "A_2", True, [0] * 10, [1] * 9 + [0], [1 / 3] * 10, [0] * 10, ["own"] * 10
    )
    m, _ = _bundle([row], chain=("W_1", "A_2"))
    tc = m["conformal"]["test_coverage"]
    assert tc["whole_test"]["meets_target"] is True
    assert (
        tc["per_session"][0]["meets_target"] is True
        and tc["n_sessions_below_target"] == 0
    )


def test_method_summary_of_one_session_has_its_gap_to_target():
    """With one session the mean gap is that session's distance to the target."""
    rows = [{"methods": {"m": _ev([1, 1, 0, 1], [0.5] * 4, [0, 0, 0, 0], ["own"] * 4)}}]
    s = method_summary(rows, "m", 0.9)
    assert s["coverage"] == pytest.approx(0.75)
    assert s["mean_abs_gap"] == pytest.approx(0.15) and s["sessions_below"] == 1
    assert s["accuracy"] == 1.0 and s["share_own"] == 1.0


def test_sigma_nets_payload_rebuilds_each_sessions_scale(tmp_path):
    """A saved payload loads back into a ScaleNet that gives the same sigma as the original."""
    import torch

    from src.conformal.learned import ScaleNet, scale_of

    torch.manual_seed(0)
    net = ScaleNet(len(SIGNALS), hidden=8, depth=1)
    with torch.no_grad():
        net.net[-1].weight.normal_()
    rows = [{"session": "A", "sigma_state": net.state_dict(),
             "signal_stats": (np.zeros(len(SIGNALS)), np.ones(len(SIGNALS)))}]  # fmt: skip
    torch.save(sigma_nets_payload(rows, hidden=8, depth=1), tmp_path / "s.pt")
    p = torch.load(tmp_path / "s.pt", weights_only=False)
    back = ScaleNet(len(p["signals"]), hidden=p["hidden"], depth=p["depth"])
    back.load_state_dict(p["sessions"][0]["state_dict"])
    z = np.random.default_rng(0).standard_normal((5, len(SIGNALS)))
    np.testing.assert_array_equal(scale_of(back, z), scale_of(net, z))


def test_sigma_nets_payload_lists_each_sessions_net_and_standardization():
    """Session i of the payload is row i, with its own weights and signal mean/std."""
    rows = [
        {
            "session": "A",
            "sigma_state": {"w": 1},
            "signal_stats": ([0.0, 1.0], [2.0, 3.0]),
        },
        {
            "session": "B",
            "sigma_state": {"w": 2},
            "signal_stats": (np.zeros(2), np.ones(2)),
        },
    ]
    p = sigma_nets_payload(rows, hidden=16, depth=3)
    assert p["signals"] == list(SIGNALS) and (p["hidden"], p["depth"]) == (16, 3)
    assert [s["session"] for s in p["sessions"]] == ["A", "B"]
    assert p["sessions"][0] == {
        "session": "A", "state_dict": {"w": 1}, "signal_mean": [0.0, 1.0],
        "signal_std": [2.0, 3.0],
    }  # fmt: skip
    assert p["sessions"][1]["signal_std"] == [1.0, 1.0]
