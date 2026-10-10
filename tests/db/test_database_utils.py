"""Tests for src/db/database_utils.py.

Accepted gaps: none. The resolver's every branch is exercised against an in-memory stand-in
for the ``subjects`` and ``dates`` collections.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from src.db.database_utils import session_name_to_date_id, session_name_to_display


def _mock_db():
    """A MagicMock db where each collection name gets its own child mock."""
    db = MagicMock()
    _collections: dict = {}
    db.__getitem__ = MagicMock(
        side_effect=lambda key: _collections.setdefault(key, MagicMock())
    )
    return db


def _db_with_dates(dates, subject=True):
    """A db whose mouse exists (or not) and owns the given date documents."""
    db = _mock_db()
    db["subjects"].find_one.return_value = {"_id": ObjectId()} if subject else None
    db["dates"].find.return_value = dates
    return db


def test_multiple_matches_returns_exact_session_path_first():
    """Resolving to the wrong session corrupts every downstream number invisibly:
    the run succeeds, on another animal's data. When two date documents both
    contain the date string but only one carries the full M<id>_<YYMMDD> pattern,
    the exact match must win."""
    ambiguous_id, exact_id = ObjectId(), ObjectId()
    db = _db_with_dates(
        [
            {"_id": ambiguous_id, "pos_path": "/archive/250214_partial/pos.csv"},
            {"_id": exact_id, "pos_path": "/data/M21_250214/pos.csv"},
        ]
    )
    assert session_name_to_date_id(db, "M21_250214") == exact_id


def test_mouse_id_is_parsed_as_an_integer():
    """The subjects lookup uses the numeric mouse id, so "M07" and "M7" name the same mouse."""
    only = ObjectId()
    db = _db_with_dates([{"_id": only, "pos_path": "/d/M7_240918/pos.csv"}])
    session_name_to_date_id(db, "M07_240918")
    db["subjects"].find_one.assert_called_once_with({"mouse_id": 7})


def test_single_match_is_returned():
    """Of several dates of one mouse, the only one whose path holds the date is chosen."""
    wanted = ObjectId()
    db = _db_with_dates(
        [
            {"_id": ObjectId(), "pos_path": "/d/M19_240917/pos.csv"},
            {"_id": wanted, "pos_path": "/d/M19_240918/pos.csv"},
            {"_id": ObjectId(), "pos_path": None},
        ]
    )
    assert session_name_to_date_id(db, "M19_240918") == wanted


def test_dotted_date_folders_are_matched():
    """Some recordings store the date as YY.MM.DD in the path; that is the same session."""
    wanted = ObjectId()
    db = _db_with_dates(
        [
            {"_id": ObjectId(), "pos_path": "/d/24.09.17/pos.csv"},
            {"_id": wanted, "pos_path": "/d/24.09.18/pos.csv"},
        ]
    )
    assert session_name_to_date_id(db, "M19_240918") == wanted


def test_ambiguous_match_without_an_exact_path_warns_and_takes_the_first(capsys):
    """With no exact M<id>_<YYMMDD> path to break the tie, the first match is used, loudly."""
    first, second = ObjectId(), ObjectId()
    db = _db_with_dates(
        [
            {"_id": first, "pos_path": "/a/240918/pos.csv"},
            {"_id": second, "pos_path": "/b/24.09.18/pos.csv"},
        ]
    )
    assert session_name_to_date_id(db, "M19_240918") == first
    assert "multiple dates match" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "db_kwargs", "message"),
    [
        ("19_240918", {"dates": []}, "Invalid session_name"),
        ("M19-240918", {"dates": []}, "Invalid session_name"),
        ("M19_240918_x", {"dates": []}, "Invalid session_name"),
        ("M19_240918", {"dates": [], "subject": False}, "not found in subjects"),
        ("M19_240918", {"dates": []}, "No dates found"),
        (
            "M19_240918",
            {"dates": [{"_id": ObjectId(), "pos_path": "/d/M19_240917/p.csv"}]},
            "Could not find session",
        ),
    ],
)
def test_unresolvable_names_raise(name, db_kwargs, message):
    """A malformed name, an unknown mouse, a mouse with no dates, or no date match all raise
    instead of returning some other session's id."""
    db = _db_with_dates(**db_kwargs)
    with pytest.raises(ValueError, match=message):
        session_name_to_date_id(db, name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("M19_240918", ("19", "2024-09-18")),
        ("m5_250101", ("5", "2025-01-01")),
        ("M19_2409", ("19", "2409")),  # not six digits: left as it is
        ("M19", ("19", "")),
    ],
)
def test_session_name_to_display(name, expected):
    """A session name reads as its mouse id and ISO date; an unexpected date is passed through."""
    assert session_name_to_display(name) == expected
