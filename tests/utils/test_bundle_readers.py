"""Tests for src/utils/bundle_readers.py."""

from __future__ import annotations

import pytest

from src.utils.bundle_readers import (
    maze_summary_table,
    per_region_coverage_table,
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
                    {"coverage": 0.93, "meets_target": True},
                    {"coverage": 0.81, "meets_target": False},
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
    b = _section_bundle()
    rows = maze_summary_table(
        {"More_Water": b, "Y_Maze": _section_bundle()}, ["Y_Maze", "More_Water"]
    )
    assert [r["maze"] for r in rows] == ["Y_Maze", "More_Water"]
    row = rows[1]
    assert row["test_cov"] == pytest.approx(0.88)
    assert row["n_mice"] == 2 and row["n_sessions"] == 3
    assert row["n_classes"] == 8 and row["raw_test_acc"] == pytest.approx(0.42)
    assert row["size_over_k"] == pytest.approx(4.0 / 8)
