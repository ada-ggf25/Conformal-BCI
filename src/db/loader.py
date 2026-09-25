"""Raw data loaders for the functional_atlas GridFS collections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bson
import zstandard
from bson import ObjectId

if TYPE_CHECKING:
    from pymongo.database import Database


def _read_gridfs(db: Database, collection: str, file_id: ObjectId) -> dict:
    """Reassemble one GridFS file."""
    chunks = list(db[f"{collection}.chunks"].find({"files_id": file_id}).sort("n", 1))
    raw = b"".join(c["data"] for c in chunks)
    decompressed = zstandard.ZstdDecompressor().decompress(
        raw, max_output_size=200_000_000
    )
    return bson.decode_all(decompressed)[0]


def get_behavior(db: Database, trial_id: ObjectId) -> dict[str, list]:
    """Behavioral timeseries for a trial.

    Keys: channel names ('X_Position', 'Speed', 'Can_See_Water', ...).
    Values: aligned lists of float or bool sampled at ~90 Hz.
    Returns: {} if no file is found for the given trial_id.
    """
    f = db["behavior_data.files"].find_one({"metadata.trial_db_id": trial_id})
    if f is None:
        return {}
    return _read_gridfs(db, "behavior_data", f["_id"])


def get_neural(db: Database, trial_id: ObjectId) -> dict[str, list[float]]:
    """Spike times for all neurons recorded in a trial.

    Keys: neuron ObjectId strings (str(neurons._id)).
    Values: lists of float sample indices relative to the session start.
    Returns: {} if no file is found for the given trial_id.
    """
    f = db["neurons_data.files"].find_one({"metadata.trial_db_id": trial_id})
    if f is None:
        return {}
    return _read_gridfs(db, "neurons_data", f["_id"])
