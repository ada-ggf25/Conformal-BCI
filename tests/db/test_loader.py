"""Tests for src/db/loader.py.

Accepted gaps: none. The chunks are real zstd-compressed BSON, so decoding is exercised for real.
"""

from __future__ import annotations

import bson
import zstandard
from bson import ObjectId

from src.db.loader import _read_gridfs, get_behavior, get_neural


class _FakeCursor:
    def __init__(self, docs: list[dict]):
        """Store the documents this cursor will sort."""
        self._docs = docs

    def sort(self, key: str, direction: int) -> list[dict]:
        """Return the documents sorted by key, honoring the direction flag."""
        return sorted(self._docs, key=lambda d: d[key], reverse=direction < 0)


class _FakeCollection:
    """Stands in for a pymongo collection: ``find(...).sort(...)`` really sorts."""

    def __init__(
        self, chunks: list[dict] | None = None, files: list[dict] | None = None
    ):
        """Store the chunk documents (``.chunks``) or file documents (``.files``) it holds."""
        self._chunks = chunks or []
        self._files = files or []

    def find(self, query: dict) -> _FakeCursor:
        """Return a cursor over the chunks belonging to the requested file id."""
        return _FakeCursor(
            [c for c in self._chunks if c["files_id"] == query["files_id"]]
        )

    def find_one(self, query: dict) -> dict | None:
        """Return the file document whose metadata names the requested trial, if any."""
        want = query["metadata.trial_db_id"]
        return next(
            (f for f in self._files if f["metadata"]["trial_db_id"] == want), None
        )


def _chunked(payload: dict, file_id: ObjectId, n_chunks: int = 4) -> list[dict]:
    """Real zstd-compressed BSON split into ``n_chunks``, returned in reverse order."""
    blob = zstandard.ZstdCompressor().compress(bson.encode(payload))
    step = len(blob) // n_chunks + 1
    pieces = [blob[i : i + step] for i in range(0, len(blob), step)]
    docs = [{"n": i, "data": p, "files_id": file_id} for i, p in enumerate(pieces)]
    return list(reversed(docs))


def test_reassembles_chunks_stored_out_of_order():
    """Joining in storage order rather than ``n`` order yields a corrupt frame that
    zstd cannot decode, so a payload that round-trips is proof of the ordering."""
    file_id = ObjectId()
    payload = {"field": list(range(50)), "name": "session"}
    db = {"mycol.chunks": _FakeCollection(chunks=_chunked(payload, file_id))}
    assert _read_gridfs(db, "mycol", file_id) == payload


def test_only_the_requested_files_chunks_are_joined():
    """Chunks of another file in the same bucket must not be spliced into this one."""
    mine, other = ObjectId(), ObjectId()
    payload = {"x": [1.0, 2.0]}
    chunks = _chunked(payload, mine) + _chunked({"y": list(range(99))}, other)
    db = {"b.chunks": _FakeCollection(chunks=chunks)}
    assert _read_gridfs(db, "b", mine) == payload


def _two_bucket_db(trial: ObjectId) -> dict:
    """A db holding one behavior file and one neural file, both for ``trial``."""
    beh_id, neu_id = ObjectId(), ObjectId()
    meta = {"metadata": {"trial_db_id": trial}}
    return {
        "behavior_data.files": _FakeCollection(files=[{"_id": beh_id, **meta}]),
        "behavior_data.chunks": _FakeCollection(
            chunks=_chunked({"X_Position": [1.0, 2.0]}, beh_id)
        ),
        "neurons_data.files": _FakeCollection(files=[{"_id": neu_id, **meta}]),
        "neurons_data.chunks": _FakeCollection(
            chunks=_chunked({"n1": [100.0, 250.0]}, neu_id)
        ),
    }


def test_behavior_and_neural_read_their_own_buckets():
    """Behavior comes from behavior_data and spikes from neurons_data; swapping the buckets
    would feed positions to the encoder as spike times."""
    trial = ObjectId()
    db = _two_bucket_db(trial)
    assert get_behavior(db, trial) == {"X_Position": [1.0, 2.0]}
    assert get_neural(db, trial) == {"n1": [100.0, 250.0]}


def test_a_trial_without_a_file_reads_as_empty():
    """No GridFS file for the trial is an empty dict, which callers treat as "skip this trial"."""
    db = _two_bucket_db(ObjectId())
    missing = ObjectId()
    assert get_behavior(db, missing) == {}
    assert get_neural(db, missing) == {}
