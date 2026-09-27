"""Tests for src/utils/bundle_readers.py.

Bundles are built by hand (or written by ``build_chain_bundle`` and read back), so every
expected number is worked out from the values put in.

Accepted gaps: none.
"""

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
    n_mice,
    operating_curves,
    per_region_coverage_table,
    per_session_table,
    region_distribution,
    seed_method_table,
    seed_operating_curves,
    seed_per_session_table,
    seed_session_coverage,
    seed_values,
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


def test_loader_reads_seed_directories_and_ignores_a_pre_chain_bundle(tmp_path):
    """Per-seed bundles are keyed by seed; a bundle directly under ALL_MICE/ is not read."""
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
    assert order == ["More_Water"]
    assert sorted(mazes["More_Water"]) == [42, 43]


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


def _seed_bundle(fresh_cov, sessions):
    """A minimal chain bundle: a method table and per-session rows (session, coverage, error)."""
    b = _section_bundle()
    b["conformal"]["methods"] = {
        "all": {"chain_B3": {"coverage": 0.9, "size_over_k": 0.5, "accuracy": 0.3,
                             "mean_abs_gap": 0.05},
                "fresh": {"coverage": fresh_cov, "size_over_k": 0.6, "accuracy": 0.2,
                          "mean_abs_gap": 0.10}}
    }  # fmt: skip
    b["conformal"]["test_coverage"]["per_session"] = [
        {"session": s, "coverage": c, "error": e, "first_of_mouse": f, "n_neurons": 9}
        for s, c, e, f in sessions
    ]
    return b


def test_method_table_is_mean_and_sd_across_seeds():
    """Each method's field is averaged over seeds; a field equal in every seed has sd 0."""
    runs = {42: _seed_bundle(0.80, []), 43: _seed_bundle(0.84, [])}
    t = seed_method_table(runs)
    assert t["fresh"]["coverage"][0] == pytest.approx(0.82)
    assert t["fresh"]["coverage"][1] == pytest.approx(np.std([0.80, 0.84], ddof=1))
    assert t["chain_B3"]["coverage"] == pytest.approx((0.9, 0.0))


def test_per_session_table_matches_sessions_by_name_in_chain_order():
    """The second seed lists the sessions in another order; results pair up by name."""
    a = [("M1_240102", 0.8, 0.5, True), ("M2_240103", 1.0, 0.7, False)]
    runs = {
        42: _seed_bundle(0.8, a),
        43: _seed_bundle(0.8, [(n, c - 0.2, e, f) for n, c, e, f in a][::-1]),
    }
    t = seed_per_session_table(runs)
    assert t["sessions"] == ["M1_240102", "M2_240103"]
    assert t["coverage"] == pytest.approx([0.7, 0.9])
    assert t["accuracy"] == pytest.approx([0.5, 0.3])
    assert t["first_of_mouse"] == [True, False]


# --------------------------------------------------------------------------- single-bundle readers


def test_n_mice_counts_distinct_mouse_prefixes():
    """Three sessions of two mice is two mice; a bundle with no sessions has none."""
    assert n_mice(_section_bundle()) == 2
    assert n_mice({}) == 0


def test_region_distribution_is_sorted_and_normalized():
    """Counts are ordered by region number and divided by their total."""
    r = region_distribution(_section_bundle())
    assert r["regions"] == [0, 1, 2] and r["counts"] == [60, 10, 30]
    assert r["fraction"] == pytest.approx([0.6, 0.1, 0.3])


def test_region_distribution_with_no_windows_is_nan():
    """A zero total gives NaN fractions, not a division error or zeros."""
    r = region_distribution({"class_counts": {"0": 0, "1": 0}})
    assert np.isnan(r["fraction"]).all()


def test_operating_curve_is_sorted_by_coverage_and_normalized_by_k():
    """The frontier is drawn left to right in coverage; sizes are divided by K."""
    b = _section_bundle()
    b["conformal"]["retrain_sweep"] = {
        "learned": [
            {"target": 0.9, "coverage": 0.85, "size": 6.0},
            {"target": 0.5, "coverage": 0.45, "size": 2.0},
        ]
    }
    oc = operating_curves(b)
    assert oc["coverage"] == [0.45, 0.85] and oc["target"] == [0.5, 0.9]
    assert oc["size_over_k"] == pytest.approx([0.25, 0.75])
    assert oc["coverage_target"] == pytest.approx(0.9) and oc["n_classes"] == 8


def test_operating_curve_without_a_class_count_has_nan_sizes():
    """Without K the normalized size is undefined, and an absent sweep is empty."""
    b = {
        "conformal": {
            "retrain_sweep": {"learned": [{"target": 0.9, "coverage": 1, "size": 2}]}
        }
    }
    assert np.isnan(operating_curves(b)["size_over_k"][0])
    assert operating_curves({})["coverage"] == []


def test_deployed_point_without_the_arm_is_nan():
    """A bundle missing the deployed arm reports NaN size, never 0."""
    b = _section_bundle()
    del b["conformal"]["arms"]
    assert np.isnan(deployed_operating_point(b)["size_over_k"])
    assert deployed_operating_point(_section_bundle())["size_over_k"] == pytest.approx(
        0.5
    )


def test_sessions_below_target_is_recounted_when_not_stored():
    """Older bundles lack the stored count; it is then the per-session misses."""
    b = _section_bundle()
    del b["conformal"]["test_coverage"]["n_sessions_below_target"]
    b["conformal"]["test_coverage"]["per_session"].append({"coverage": 0.5})
    w = whole_test_coverage(b)
    assert w["n_sessions_below"] == 2  # one stored False, one missing flag
    assert w["per_session_meets"] == [True, False, False]


def test_empty_bundle_reads_as_nan_not_zero():
    """A bundle with no conformal section reads NaN coverage and the default 90% target."""
    w = whole_test_coverage({})
    assert np.isnan(w["coverage"]) and w["n_windows"] == 0 and w["target"] == 0.9


def test_per_region_size_is_none_where_not_reported():
    """A region with no size keeps None rather than a fake 0."""
    assert per_region_coverage_table(_section_bundle())["size_over_k"][1:] == [
        None,
        None,
    ]


def test_method_and_session_tables_default_to_empty():
    """A bundle without the chain sections yields empty tables."""
    assert method_table({}) == {} and per_session_table({}) == []
    assert method_table(_seed_bundle(0.8, []), "first") == {}


def test_error_correlations_need_saved_arrays():
    """Without arrays.npz there is nothing to correlate."""
    assert error_correlations_for_bundle(_section_bundle()) is None


# --------------------------------------------------------------------------- seed aggregation


def test_seed_values_are_in_seed_order_and_drop_none():
    """Values come out sorted by seed; a reader returning None for a seed is skipped."""
    runs = {44: {"v": 3}, 42: {"v": 1}, 43: {"v": None}}
    assert seed_values(runs, lambda b: b["v"]) == [1, 3]


def test_mean_sd_of_nothing_is_nan():
    """No seeds at all gives a NaN mean and sd."""
    mean, sd = mean_sd([])
    assert np.isnan(mean).all() and np.isnan(sd).all()


def test_mean_sd_of_scalars_matches_numpy():
    """For plain scalars it is the ordinary mean and sample sd."""
    mean, sd = mean_sd([1.0, 2.0, 6.0])
    assert mean[0] == pytest.approx(3.0)
    assert sd[0] == pytest.approx(np.std([1.0, 2.0, 6.0], ddof=1))


def test_maze_summary_skips_unknown_mazes():
    """A maze named in the order but not loaded produces no row."""
    rows = maze_summary_table(
        {"Y_Maze": {1: _section_bundle()}}, ["Nested_All", "Y_Maze"]
    )
    assert [r["maze"] for r in rows] == ["Y_Maze"]


def test_method_missing_from_one_seed_is_averaged_over_the_others():
    """A method only one seed ran is reported from that seed, with no spread."""
    a, b = _seed_bundle(0.8, []), _seed_bundle(0.9, [])
    del b["conformal"]["methods"]["all"]["fresh"]
    t = seed_method_table({42: a, 43: b})
    assert t["fresh"]["coverage"][0] == pytest.approx(0.8)
    assert np.isnan(t["fresh"]["coverage"][1])


def test_seed_method_table_of_bundles_without_methods_is_empty():
    """Pre-chain bundles have no method table to aggregate."""
    assert seed_method_table({42: _section_bundle()}) == {}


def test_a_session_missing_from_one_seed_counts_as_missing():
    """A session absent from a seed is averaged over the seeds that have it."""
    a = [("M1_1", 0.8, 0.5, True), ("M2_1", 1.0, 0.7, False)]
    runs = {42: _seed_bundle(0.8, a), 43: _seed_bundle(0.8, a[:1])}
    t = seed_per_session_table(runs)
    assert t["coverage"] == pytest.approx([0.8, 1.0])
    assert t["coverage_sd"][0] == pytest.approx(0.0) and np.isnan(t["coverage_sd"][1])
    assert t["n_neurons"] == [9, 9] and t["target"] == pytest.approx(0.9)


def test_a_sweep_target_missing_from_one_seed_is_not_zero():
    """A target one seed lacks is averaged over the seeds that swept it."""
    b1, b2 = _section_bundle(), _section_bundle()
    b1["conformal"]["retrain_sweep"] = {
        "learned": [{"target": 0.5, "coverage": 0.4, "size": 2.0},
                    {"target": 0.9, "coverage": 0.8, "size": 4.0}]
    }  # fmt: skip
    b2["conformal"]["retrain_sweep"] = {
        "learned": [{"target": 0.9, "coverage": 0.9, "size": 4.0}]
    }
    oc = seed_operating_curves({42: b1, 43: b2})
    assert oc["coverage"] == pytest.approx([0.4, 0.85])


def test_operating_curves_without_any_sweep_are_empty():
    """No seed swept anything: the curve is empty."""
    oc = seed_operating_curves({42: _section_bundle()})
    assert oc["target"] == [] and oc["coverage"] == []


# --------------------------------------------------------------------------- loading


def _write(path, bundle):
    """Write one metrics.json under path, creating it."""
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps(bundle))


def test_loader_skips_incomplete_bundles_and_orders_mazes(tmp_path):
    """A bundle without test coverage is skipped with a message; mazes follow the preferred
    order, then any others alphabetically."""
    _write(tmp_path / "Z" / "ALL_MICE" / "seed_1", {**_section_bundle(), "paradigm": "Zeta",
                                                    "seed": 1})  # fmt: skip
    _write(tmp_path / "A" / "ALL_MICE" / "seed_1", {**_section_bundle(), "paradigm": "Alpha",
                                                    "seed": 1})  # fmt: skip
    _write(tmp_path / "Y" / "ALL_MICE" / "seed_1", {**_section_bundle(), "paradigm": "Y_Maze",
                                                    "seed": 1})  # fmt: skip
    _write(tmp_path / "Y" / "ALL_MICE" / "seed_2", {"paradigm": "Y_Maze", "seed": 2,
                                                    "conformal": {}})  # fmt: skip
    logs: list[str] = []
    mazes, order = load_maze_bundles(tmp_path, log_fn=logs.append)
    assert order == ["Y_Maze", "Alpha", "Zeta"]
    assert list(mazes["Y_Maze"]) == [1]
    assert len(logs) == 1 and "seed_2" in logs[0]
    assert load_maze_bundles(tmp_path)[1] == order  # the log sink is optional


def test_loader_records_which_sidecar_files_exist(tmp_path):
    """The arrays and sigma-net paths are attached only when those files were saved."""
    d = tmp_path / "Y_Maze" / "ALL_MICE" / "seed_42"
    _write(d, {**_section_bundle(), "seed": 42})
    (d / "arrays.npz").write_bytes(b"")
    b = load_maze_bundles(tmp_path)[0]["More_Water"][42]
    assert b["_arrays_path"] == d / "arrays.npz"
    assert b["_sigma_nets_path"] is None


def test_paradigm_falls_back_to_the_directory_name(tmp_path):
    """A bundle without a paradigm field is filed under its directory."""
    b = {**_section_bundle(), "seed": 3}
    del b["paradigm"]
    _write(tmp_path / "Y_Maze" / "ALL_MICE" / "seed_3", b)
    assert list(load_maze_bundles(tmp_path)[0]) == ["Y_Maze"]
