"""Tests for src/utils/discovery.py.

MongoDB is an in-memory fake that answers ``find`` / ``find_one`` by field equality, and the
neuron count per session is patched, so which sessions and paradigm come out is decided by hand.

Accepted gaps: none.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from bson import ObjectId

from src.data.config import MIN_NEURONS_PER_SESSION
from src.utils.discovery import get_all_mice, get_contexts_for_mouse, resolve_paradigm


class _Collection:
    """A list of documents queried by equality on (dotted) fields."""

    def __init__(self, docs: list[dict]):
        """Store the documents."""
        self.docs = docs

    @staticmethod
    def _get(doc: dict, dotted: str):
        """Read a dotted field, or None if a part is missing."""
        for part in dotted.split("."):
            if not isinstance(doc, dict) or part not in doc:
                return None
            doc = doc[part]
        return doc

    def find(self, query: dict) -> list[dict]:
        """Every document matching all of the query's fields."""
        return [
            d for d in self.docs if all(self._get(d, k) == v for k, v in query.items())
        ]

    def find_one(self, query: dict) -> dict | None:
        """The first matching document, or None."""
        found = self.find(query)
        return found[0] if found else None


class _DB(dict):
    """A database: collection name -> _Collection, with the ``name`` attribute pymongo has."""

    name = "fake_db"

    def __missing__(self, key):
        """An unknown collection is empty."""
        return _Collection([])


def _db(sessions: list[tuple[int, str, str, bool, bool]]) -> _DB:
    """Build a db from (mouse, paradigm, pos_path, has_trial, has_neural_file) rows."""
    mice = sorted({m for m, *_ in sessions})
    subj = {m: ObjectId() for m in mice}
    dates, trials, files = [], [], []
    for mouse, paradigm, path, has_trial, has_neural in sessions:
        date_id = ObjectId()
        dates.append(
            {
                "_id": date_id,
                "mouse_db_id": subj[mouse],
                "paradigm": paradigm,
                "pos_path": path,
            }
        )
        if has_trial:
            tid = ObjectId()
            trials.append({"_id": tid, "date_db_id": date_id})
            if has_neural:
                files.append({"_id": ObjectId(), "metadata": {"trial_db_id": tid}})
    return _DB(
        subjects=_Collection(
            [{"_id": subj[m], "mouse_id": m} for m in mice] + [{"_id": ObjectId()}]
        ),
        dates=_Collection(dates),
        trials=_Collection(trials),
        **{"neurons_data.files": _Collection(files)},
    )


def test_all_mice_skips_subjects_without_an_id():
    """Every subject with a mouse id is listed; a subject document without one is ignored."""
    db = _db(
        [(3, "Y_Maze", "/M3_240101", True, True), (1, "Y_Maze", "/M1_240101", 1, 1)]
    )
    assert sorted(get_all_mice(db)) == [1, 3]


class TestGetContextsForMouse:
    """Which of a mouse's dates count as sessions, by paradigm."""

    def test_only_sessions_with_a_trial_and_spikes_are_listed(self):
        """A date needs its own M<id>_<YYMMDD> path, a trial, and a spike file to count."""
        db = _db(
            [
                (1, "Y_Maze", "/data/M1_240103/pos.csv", True, True),
                (1, "Y_Maze", "/data/M1_240101/pos.csv", True, True),
                (
                    1,
                    "Y_Maze",
                    "/data/M1_240101/pos2.csv",
                    True,
                    True,
                ),  # duplicate session
                (1, "Y_Maze", "/data/M1_240102/pos.csv", True, False),  # no spike file
                (1, "Y_Maze", "/data/M1_240104/pos.csv", False, False),  # no trial
                (
                    1,
                    "Y_Maze",
                    "/data/M11_240105/pos.csv",
                    True,
                    True,
                ),  # another mouse's id
                (1, "Y_Maze", "/data/240106/pos.csv", True, True),  # no session name
                (1, "More_Water", "/data/M1_240201/pos.csv", True, True),
            ]
        )
        assert get_contexts_for_mouse(db, 1) == {
            "Y_Maze": ["M1_240101", "M1_240103"],
            "More_Water": ["M1_240201"],
        }

    def test_unknown_mouse_has_no_sessions(self):
        """A mouse absent from subjects has no contexts."""
        assert get_contexts_for_mouse(_db([]), 99) == {}

    def test_missing_paradigm_is_grouped_as_unknown(self):
        """A date without a paradigm field is still listed, under "Unknown"."""
        db = _db([(1, "Y_Maze", "/M1_240101", True, True)])
        del db["dates"].docs[0]["paradigm"]
        assert get_contexts_for_mouse(db, 1) == {"Unknown": ["M1_240101"]}


def _neurons(counts: dict[str, int]):
    """Patch the neuron lookup so each session has the given number of units."""
    return patch(
        "src.utils.discovery.get_session_neuron_ids",
        side_effect=lambda _db, s: [f"n{i}" for i in range(counts.get(s, 0))],
    )


def _contexts(contexts: dict[str, list[str]]):
    """Patch the context lookup to return the given paradigm -> sessions map."""
    return patch("src.utils.discovery.get_contexts_for_mouse", return_value=contexts)


class TestResolveParadigm:
    """Which paradigm a mouse is analyzed under, and its valid sessions."""

    def test_the_paradigm_with_most_valid_sessions_wins(self):
        """Y has three sessions but only one with enough neurons; Linear has two valid ones."""
        ctx = {
            "Y_Maze": ["M1_240103", "M1_240101", "M1_240102"],
            "More_Water": ["M1_240202", "M1_240201"],
        }
        counts = {"M1_240101": 9, "M1_240201": 9, "M1_240202": 9}
        with _contexts(ctx), _neurons(counts):
            assert resolve_paradigm(None, 1) == (
                "More_Water",
                ["M1_240201", "M1_240202"],
            )

    def test_a_tie_keeps_the_first_paradigm_listed(self):
        """Two paradigms with equally many valid sessions: the first one found is kept."""
        ctx = {"Y_Maze": ["a", "b"], "More_Water": ["c", "d"]}
        with _contexts(ctx), _neurons(dict.fromkeys("abcd", 9)):
            assert resolve_paradigm(None, 1)[0] == "Y_Maze"

    def test_target_paradigm_restricts_the_choice(self):
        """With a target, only that paradigm is considered even if another has more sessions."""
        ctx = {"Y_Maze": ["a", "b", "c"], "More_Water": ["d", "e"]}
        with _contexts(ctx), _neurons(dict.fromkeys("abcde", 9)):
            assert resolve_paradigm(None, 1, "More_Water") == ("More_Water", ["d", "e"])

    def test_default_neuron_floor_is_the_configured_minimum(self):
        """A session exactly at the minimum is valid; one unit fewer is not."""
        n = MIN_NEURONS_PER_SESSION
        with _contexts({"Y": ["a", "b", "c"]}), _neurons({"a": n, "b": n, "c": n - 1}):
            assert resolve_paradigm(None, 1) == ("Y", ["a", "b"])

    def test_explicit_neuron_floor(self):
        """A caller-supplied floor replaces the default."""
        with _contexts({"Y": ["a", "b", "c"]}), _neurons({"a": 2, "b": 3, "c": 3}):
            assert resolve_paradigm(None, 1, min_neurons=3) == ("Y", ["b", "c"])

    @pytest.mark.parametrize(
        ("ctx", "counts", "target", "message"),
        [
            ({}, {}, "", "No sessions found"),
            ({"Y": ["a"]}, {"a": 9}, "", "no paradigm with >=2"),
            ({"Y": ["a", "b"]}, {"a": 9}, "", "no paradigm with >=2"),
            ({"Y": ["a", "b"]}, {"a": 9, "b": 9}, "Nested_Maze", "TARGET_PARADIGM"),
        ],
    )
    def test_a_mouse_without_two_valid_sessions_raises(
        self, ctx, counts, target, message
    ):
        """No sessions, a single session, too few neurons, or a missing target all raise."""
        with (
            _contexts(ctx),
            _neurons(counts),
            pytest.raises(RuntimeError, match=message),
        ):
            resolve_paradigm(_DB(), 1, target)
