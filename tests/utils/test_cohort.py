"""Tests for src/utils/cohort.py.

``get_all_mice`` and ``resolve_paradigm`` are patched, so each mouse's sessions (or its reason
for being skipped) are chosen by hand and the pooled cohort is checked against them.

Accepted gaps: ``MERGED`` itself (a literal preset table) is only exercised through the
``Nested_All`` expansion; and whether ``paradigm_label`` applies to a single paradigm (the
docstring says it is ignored, the code applies it) is left unpinned until that is settled.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.data.regions import get_num_classes
from src.utils.cohort import (
    apply_session_exclusions,
    assert_single_family,
    discover_cohort,
)


class TestAssertSingleFamily:
    def test_one_family_passes(self):
        """The two Nested blocks and a Nested label are one family."""
        assert_single_family(["Nested_Maze", "Nested_Maze+2"], "Nested_All")

    @pytest.mark.parametrize(
        ("paradigms", "label"),
        [
            (["Y_Maze", "Nested_Maze"], "Y_Maze"),
            (["More_Water", "Y_Maze"], "More_Water"),
            (
                ["Nested_Maze"],
                "Y_Maze",
            ),  # members agree, label routes to the wrong geometry
        ],
    )
    def test_mixed_families_are_refused(self, paradigms, label):
        """Pooling across geometries, or mislabeling a pool, would label regions wrongly."""
        with pytest.raises(AssertionError, match="cannot pool"):
            assert_single_family(paradigms, label)


def _resolver(table: dict[tuple[int, str], list[str] | str]):
    """Patch resolve_paradigm: a list is the mouse's sessions, a string is the error raised."""

    def resolve(_db, mouse, paradigm):
        """Return the sessions for (mouse, paradigm), or raise its RuntimeError."""
        out = table.get((mouse, paradigm), f"Mouse {mouse}: nothing\nsecond line")
        if isinstance(out, str):
            raise RuntimeError(out)
        return paradigm, out

    return patch("src.utils.cohort.resolve_paradigm", side_effect=resolve)


def _mice(ids: list[int]):
    """Patch get_all_mice to return the given ids (deliberately unsorted)."""
    return patch("src.utils.cohort.get_all_mice", return_value=ids)


class TestDiscoverCohort:
    def test_merged_preset_pools_both_blocks(self):
        """Nested_All pools Nested_Maze and Nested_Maze+2; a mouse may contribute through
        either block and each session remembers which block it came from."""
        table = {
            (1, "Nested_Maze"): ["M1_240102", "M1_240101"],
            (2, "Nested_Maze+2"): ["M2_240105", "M2_240106"],
            (1, "Nested_Maze+2"): ["M1_240110", "M1_240111"],
        }
        with _mice([2, 3, 1]), _resolver(table):
            c = discover_cohort(None, "Nested_All")
        assert c.paradigms == ["Nested_Maze", "Nested_Maze+2"]
        assert c.label == "Nested_All"
        assert c.n_classes == get_num_classes("Nested_All") == 9
        assert c.group == sorted(
            [
                "M1_240101",
                "M1_240102",
                "M1_240110",
                "M1_240111",
                "M2_240105",
                "M2_240106",
            ]
        )
        assert c.cohort_mice == [1, 2]  # get_all_mice is sorted first
        assert c.session_to_mouse["M2_240105"] == 2
        assert c.session_to_paradigm["M1_240110"] == "Nested_Maze+2"
        assert c.session_to_paradigm["M1_240101"] == "Nested_Maze"

    def test_skipped_mice_record_the_first_line_of_each_reason(self):
        """A mouse with nothing usable is reported with one reason per paradigm tried."""
        with _mice([3]), _resolver({(4, "Y_Maze"): ["a", "b"]}):
            c = discover_cohort(None, "Y_Maze", target_mice=[3, 4])
        assert c.cohort_mice == [4]
        assert c.skipped == [(3, "Y_Maze: Mouse 3: nothing")]

    def test_target_and_excluded_mice(self):
        """target_mice replaces the full list (get_all_mice is not consulted); exclusions apply."""
        table = {(m, "Y_Maze"): [f"M{m}_1", f"M{m}_2"] for m in (1, 2, 3)}
        with _mice([]) as all_mice, _resolver(table):
            c = discover_cohort(
                None, "Y_Maze", target_mice=[3, 1, 2], exclude_mice=(1,)
            )
        all_mice.assert_not_called()
        assert c.cohort_mice == [3, 2]

    @pytest.mark.parametrize(
        ("target", "label_arg", "label"),
        [
            ("Y_Maze", "", "Y_Maze"),
            (["Nested_Maze"], "", "Nested_Maze"),
            (["Nested_Maze", "Nested_Maze+2"], "", "Nested_Maze+Nested_Maze+2"),
            (["Nested_Maze", "Nested_Maze+2"], "Nested_Pool", "Nested_Pool"),
        ],
    )
    def test_label(self, target, label_arg, label):
        """A single paradigm labels itself, a list joins with "+", an explicit label wins."""
        paradigms = [target] if isinstance(target, str) else target
        table = {(1, p): [f"M1_{i}{p}" for i in range(2)] for p in paradigms}
        with _mice([1]), _resolver(table):
            assert (
                discover_cohort(None, target, paradigm_label=label_arg).label == label
            )

    def test_a_label_from_another_family_is_refused(self):
        """A free-text label naming another maze would route the labeler to wrong geometry."""
        with _mice([1]), pytest.raises(AssertionError, match="cannot pool"):
            discover_cohort(None, ["Nested_Maze"], paradigm_label="Y_Maze_pool")

    def test_no_usable_mouse_is_an_error(self):
        """An empty cohort cannot be trained on, so discovery raises."""
        with (
            _mice([1, 2]),
            _resolver({}),
            pytest.raises(RuntimeError, match="No usable"),
        ):
            discover_cohort(None, "Y_Maze")

    def test_a_session_claimed_by_two_mice_is_refused(self):
        """Pooling the same session twice would double-count it."""
        table = {(1, "Y_Maze"): ["S_1", "S_2"], (2, "Y_Maze"): ["S_2", "S_3"]}
        with (
            _mice([1, 2]),
            _resolver(table),
            pytest.raises(AssertionError, match="duplicate"),
        ):
            discover_cohort(None, "Y_Maze")

    def test_cross_family_list_is_refused_before_touching_the_db(self):
        """The family check runs first, so no mouse is resolved for an invalid pool."""
        with _mice([1]) as all_mice, pytest.raises(AssertionError):
            discover_cohort(None, ["Y_Maze", "More_Water"])
        all_mice.assert_not_called()


class TestApplySessionExclusions:
    def test_drops_listed_sessions_and_keeps_order(self):
        """Excluded sessions are removed; the rest keep their original order."""
        logs: list[str] = []
        kept = apply_session_exclusions(["c", "a", "b"], ["a"], log_fn=logs.append)
        assert kept == ["c", "b"]
        assert len(logs) == 1 and "dropped 1" in logs[0] and "['a']" in logs[0]

    def test_unknown_exclusions_are_reported_not_fatal(self):
        """A typo in the exclusion list is logged, and drops nothing."""
        logs: list[str] = []
        kept = apply_session_exclusions(["a"], ["zz", "yy"], log_fn=logs.append)
        assert kept == ["a"]
        assert logs == ["  EXCLUDE_SESSIONS: ['yy', 'zz'] not in the cohort, ignored"]

    @pytest.mark.parametrize("exclude", [None, []])
    def test_nothing_to_exclude_keeps_everything_silently(self, exclude):
        """No exclusions: every session is kept and nothing is logged."""
        logs: list[str] = []
        assert apply_session_exclusions(["a", "b"], exclude, log_fn=logs.append) == [
            "a",
            "b",
        ]
        assert logs == []

    def test_works_without_a_log_sink(self):
        """Logging is optional."""
        assert apply_session_exclusions(["a", "b"], ["a", "q"]) == ["b"]
