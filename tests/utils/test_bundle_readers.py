"""Tests for src/utils/bundle_readers.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.artifacts import build_chain_bundle, jsonify
from src.utils.bundle_readers import (
    deployed_operating_point,
    error_correlations_for_bundle,
    load_maze_bundles,
    maze_summary_table,
    mean_sd,
    method_table,
    per_region_coverage_table,
    seed_operating_curves,
    seed_session_coverage,
    whole_test_coverage,
)


def _section_bundle() -> dict:
    """Return a bundle carrying the conformal sections the cross-maze readers consume."""
    return {
        "paradigm": "More_Water",
        "n_classes": 8,
        "sessions": ["M40_250113", "M40_250115", "M19_240918"],
        "decoder": {"raw_accuracy": 0.42},
        "conformal": {
            "alpha": 0.1,
            "test_coverage": {
                "whole_test": {
                    "coverage": 0.88,
                    "n": 1200,
                    "error": 0.55,
                    "meets_target": False,
                },
                "per_session": [
                    {"session": "M40_250113", "coverage": 0.93, "meets_target": True},
                    {"session": "M40_250115", "coverage": 0.81, "meets_target": False},
                ],
                "n_sessions_below_target": 1,
            },
            "arms": {
                "learned-scale class-conditional": {
                    "coverage": 0.88,
                    "set_size": 4.0,  # n_classes=8 -> size_over_k = 0.5
                    "size_at_matched_coverage": 4.5,
                },
            },
            "per_region_coverage": {
                # deliberately out of numeric order to prove the reader sorts
                "learned": {"0": 0.88, "2": 0.88, "1": 0.63},
                "n_test": {"0": 5911, "2": 3318, "1": 502},
                "size": {"0": 2.0, "1": None},
            },
        },
        "class_counts": {"2": 30, "0": 60, "1": 10},
    }


def test_regions_sorted_numerically_and_aligned():
    """String keys sorted as strings would put "10" before "2" and silently
    mis-pair every coverage value with the wrong region."""
    t = per_region_coverage_table(_section_bundle())
    assert t["regions"] == [0, 1, 2]
    assert t["learned"] == [0.88, 0.63, 0.88]
    assert t["n_test"] == [5911, 502, 3318]
    assert t["size_over_k"][0] == pytest.approx(2.0 / 8)


def test_whole_test_coverage_reads_headline_and_spread():
    """The headline coverage number quoted in the report."""
    w = whole_test_coverage(_section_bundle())
    assert w["coverage"] == pytest.approx(0.88)
    assert w["n_windows"] == 1200
    assert w["per_session_coverage"] == pytest.approx([0.93, 0.81])
    assert w["n_sessions"] == 2 and w["n_sessions_below"] == 1
    assert w["target"] == pytest.approx(0.9)


def test_rows_follow_order_and_compose_the_section_readers():
    """The cross-maze table itself, so this also pins n_mice and size_over_k."""
    b2 = _section_bundle()
    b2["decoder"]["raw_accuracy"] = 0.46
    rows = maze_summary_table(
        {
            "More_Water": {42: _section_bundle(), 43: b2},
            "Y_Maze": {42: _section_bundle()},
        },
        ["Y_Maze", "More_Water"],
    )
    assert [r["maze"] for r in rows] == ["Y_Maze", "More_Water"]
    row = rows[1]
    assert row["n_seeds"] == 2
    assert row["test_cov"] == pytest.approx(0.88) and row[
        "test_cov_sd"
    ] == pytest.approx(0)
    assert row["n_mice"] == 2 and row["n_sessions"] == 3
    assert row["n_classes"] == 8 and row["raw_test_acc"] == pytest.approx(0.44)
    assert row["raw_test_acc_sd"] == pytest.approx(np.std([0.42, 0.46], ddof=1))
    assert row["size_over_k"] == pytest.approx(4.0 / 8)
    assert np.isnan(rows[0]["test_cov_sd"])  # one seed: no spread


def test_mean_sd_ignores_missing_values_per_element():
    """A seed without a value for one element must not drag that element's mean to 0."""
    mean, sd = mean_sd([[1.0, None], [3.0, 5.0]])
    assert mean == pytest.approx([2.0, 5.0])
    assert sd[0] == pytest.approx(np.sqrt(2.0)) and np.isnan(sd[1])


def test_session_coverage_is_averaged_by_session_name():
    """Seeds list sessions in the same order, but matching is by name, not position."""
    b1, b2 = _section_bundle(), _section_bundle()
    b2["conformal"]["test_coverage"]["per_session"] = [
        {"session": "M40_250115", "coverage": 0.91},
        {"session": "M40_250113", "coverage": 0.83},
    ]
    out = seed_session_coverage({42: b1, 43: b2})
    got = dict(zip(out["sessions"], out["coverage"], strict=True))
    assert got == pytest.approx({"M40_250113": 0.88, "M40_250115": 0.86})
    assert out["meets"] == [False, False]


def test_operating_curves_are_matched_by_target():
    """Each target's mean comes from the same target in every seed."""

    def _with_sweep(rows):
        b = _section_bundle()
        b["conformal"]["retrain_sweep"] = {"learned": rows}
        return b

    b1 = _with_sweep(
        [
            {"target": 0.5, "coverage": 0.40, "size": 2.0},
            {"target": 0.9, "coverage": 0.80, "size": 4.0},
        ]
    )
    b2 = _with_sweep(
        [
            {"target": 0.9, "coverage": 0.90, "size": 6.0},
            {"target": 0.5, "coverage": 0.50, "size": 2.0},
        ]
    )
    oc = seed_operating_curves({42: b1, 43: b2})
    assert oc["target"] == [0.5, 0.9]
    assert oc["coverage"] == pytest.approx([0.45, 0.85])
    assert oc["size_over_k"] == pytest.approx([2.0 / 8, 5.0 / 8])


def test_loader_reads_seed_directories_and_falls_back_to_a_single_bundle(tmp_path):
    """Per-seed bundles are keyed by seed; a maze without seed dirs keeps its one bundle."""
    for seed in (43, 42):
        d = tmp_path / "More_Water" / "ALL_MICE" / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({**_section_bundle(), "seed": seed}))
    legacy = tmp_path / "Y_Maze" / "ALL_MICE"
    legacy.mkdir(parents=True)
    (legacy / "metrics.json").write_text(
        json.dumps({**_section_bundle(), "paradigm": "Y_Maze"})
    )
    mazes, order = load_maze_bundles(tmp_path)
    assert order == ["More_Water", "Y_Maze"]
    assert sorted(mazes["More_Water"]) == [42, 43]
    assert list(mazes["Y_Maze"]) == [42]


def _ev(covered, size, error, level):
    return {
        "covered": np.array(covered, float),
        "size_over_k": np.array(size, float),
        "error": np.array(error, float),
        "level": np.array(level),
    }


def test_a_written_chain_bundle_reads_back_pooled_over_windows(tmp_path):
    """Two tested sessions (2 and 3 test windows, K = 2) written by build_chain_bundle and read
    back: every headline number is pooled over the 5 test windows, worked by hand."""
    rows = []
    for name, first, labels, chain_ev, fresh_ev in (
        ("M1_240102", True, [0, 1],
         _ev([1, 0], [0.5, 1.0], [0, 1], ["own", "global"]),
         _ev([1, 1], [1.0, 1.0], [1, 1], ["global"] * 2)),
        ("M2_240103", False, [0, 0, 1],
         _ev([1, 1, 1], [0.5] * 3, [0, 0, 1], ["own"] * 3),
         _ev([0, 1, 1], [0.5, 1.0, 1.0], [0, 1, 1], ["global"] * 3)),
    ):  # fmt: skip
        n = len(labels)
        rows.append(
            {
                "session": name, "mouse": name[:2], "first_of_mouse": first, "n_neurons": 6,
                "labels": np.array(labels), "sigma_rel": np.ones(n),
                "signals": np.zeros((n, 4)),
                "methods": {"chain_B3": chain_ev, "fresh": fresh_ev},
                "sweep": {0.9: {"covered": chain_ev["covered"],
                                "size_over_k": chain_ev["size_over_k"]}},
            }
        )  # fmt: skip
    chain = ["M2_240101", "M1_240102", "M2_240103"]
    data = {
        s: {
            "regions": np.array([0, 1]),
            "train_idx": np.array([0]),
            "val_idx": np.array([1]),
            "test_idx": np.array([], int),
        }
        for s in chain
    }
    metrics, arrays = build_chain_bundle(
        rows, chain=chain, n_warmup=1, session_data=data, n_classes=2,
        paradigm="More_Water", seed=42, alpha=0.1, rule="B3", config={},
    )  # fmt: skip
    out = tmp_path / "More_Water" / "ALL_MICE" / "seed_42"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(jsonify(metrics)))
    np.savez(out / "arrays.npz", **arrays)

    b = load_maze_bundles(tmp_path)[0]["More_Water"][42]
    w = whole_test_coverage(b)
    assert w["coverage"] == pytest.approx(4 / 5)  # 1+0+1+1+1 of 5 windows
    assert w["per_session_coverage"] == pytest.approx([0.5, 1.0])
    assert w["n_sessions_below"] == 1  # 0.5 < 0.9
    assert deployed_operating_point(b)["size_over_k"] == pytest.approx(3.0 / 5)
    assert b["decoder"]["raw_accuracy"] == pytest.approx(3 / 5)
    table = method_table(b, "all")
    assert table["fresh"]["coverage"] == pytest.approx(4 / 5)
    assert table["fresh"]["accuracy"] == pytest.approx(1 / 5)
    assert table["chain_B3"]["mean_abs_gap"] == pytest.approx((0.4 + 0.1) / 2)
    assert table["chain_B3"]["share_global"] == pytest.approx(1 / 5)
    assert method_table(b, "returning")["chain_B3"]["coverage"] == pytest.approx(1.0)
    assert (
        b["conformal"]["per_region_coverage"]["learned"]["0"] is None
    )  # 3 < 5 windows
    assert error_correlations_for_bundle(b)["size_points"]["n"].sum() == 5
