"""Tests for src/db/database_utils.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from bson import ObjectId

from src.db.database_utils import session_name_to_date_id


def _mock_db():
    """A MagicMock db where each collection name gets its own child mock."""
    db = MagicMock()
    _collections: dict = {}
    db.__getitem__ = MagicMock(
        side_effect=lambda key: _collections.setdefault(key, MagicMock())
    )
    return db


def test_multiple_matches_returns_exact_session_path_first():
    """Resolving to the wrong session corrupts every downstream number invisibly:
    the run succeeds, on another animal's data. When two date documents both
    contain the date string but only one carries the full M<id>_<YYMMDD> pattern,
    the exact match must win."""
    db = _mock_db()
    ambiguous_id, exact_id = ObjectId(), ObjectId()
    db["subjects"].find_one.return_value = {"_id": ObjectId()}
    db["dates"].find.return_value = [
        {"_id": ambiguous_id, "pos_path": "/archive/250214_partial/pos.csv"},
        {"_id": exact_id, "pos_path": "/data/M21_250214/pos.csv"},
    ]
    assert session_name_to_date_id(db, "M21_250214") == exact_id
