"""Tests for src/db/loader.py."""

from __future__ import annotations

import bson
import zstandard
from bson import ObjectId

from src.db.loader import _read_gridfs


class _FakeCursor:
    def __init__(self, docs: list[dict]):
        """Store the documents this cursor will sort."""
        self._docs = docs

    def sort(self, key: str, direction: int) -> list[dict]:
        """Return the documents sorted by key, honoring the direction flag."""
        return sorted(self._docs, key=lambda d: d[key], reverse=direction < 0)


class _FakeCollection:
    """Stands in for a pymongo collection: ``find(...).sort(...)`` really sorts."""

    def __init__(self, chunks: list[dict] | None = None):
        """Store the chunk documents this collection holds."""
        self._chunks = chunks or []

    def find(self, query: dict) -> _FakeCursor:
        """Return a cursor over the chunks belonging to the requested file id."""
        return _FakeCursor(
            [c for c in self._chunks if c["files_id"] == query["files_id"]]
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
