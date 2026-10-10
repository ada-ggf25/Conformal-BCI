"""MongoDB session lookup utilities.

Session names follow the convention M<mouse_id>_<YYMMDD> (e.g. "M21_250214").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bson import ObjectId
    from pymongo.database import Database


def session_name_to_date_id(db: Database, session_name: str) -> ObjectId:
    """Resolve a session name to its MongoDB date document _id.

    The mouse's ``dates`` documents are matched on their ``pos_path``, which must contain
    the date as ``YYMMDD`` or ``YY.MM.DD``. When several match, the one whose path contains
    the full session name wins; failing that, the first match is used and a warning is
    printed.

    Args:
        db:           pymongo Database.
        session_name: e.g. ``"M21_250214"``.

    Returns:
        ObjectId of the matching document in the dates collection.

    Raises:
        ValueError: if the session name format is wrong or no match is found.
    """
    if not session_name.startswith("M"):
        raise ValueError(
            f"Invalid session_name format: {session_name!r}. Expected M<id>_<YYMMDD>."
        )

    parts = session_name.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid session_name format: {session_name!r}. Expected M<id>_<YYMMDD>."
        )

    mouse_id = int(parts[0][1:])
    date_str = parts[1]  # YYMMDD

    subject = db["subjects"].find_one({"mouse_id": mouse_id})
    if not subject:
        raise ValueError(f"Mouse {mouse_id} not found in subjects collection.")

    dates = list(db["dates"].find({"mouse_db_id": subject["_id"]}))
    if not dates:
        raise ValueError(f"No dates found for mouse {mouse_id}.")

    matched = []
    for doc in dates:
        pos_path = doc.get("pos_path") or ""
        alt = f"{date_str[:2]}.{date_str[2:4]}.{date_str[4:6]}"
        if date_str in pos_path or alt in pos_path:
            matched.append(doc)

    if len(matched) == 1:
        return matched[0]["_id"]

    if len(matched) > 1:
        for doc in matched:
            if f"M{mouse_id}_{date_str}" in doc.get("pos_path", ""):
                return doc["_id"]
        print(f"Warning: multiple dates match {session_name!r}, using first.")
        return matched[0]["_id"]

    raise ValueError(
        f"Could not find session {session_name!r} in the database "
        f"(no date matches '{date_str}' for mouse {mouse_id})."
    )


def session_name_to_display(session_name: str) -> tuple[str, str]:
    """Parse a session name into a human-readable (mouse id, date) pair, no DB lookup.

    ``"M19_240918"`` -> ``("19", "2024-09-18")``. Assumes the century is 20xx, matching
    every session in this dataset.

    Args:
        session_name: e.g. "M19_240918"

    Returns:
        ``(mouse_id, date)`` as plain strings.
    """
    mouse_part, _, date_part = session_name.partition("_")
    mouse_id = mouse_part.lstrip("Mm")
    if len(date_part) == 6:
        date = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]}"
    else:
        date = date_part
    return mouse_id, date
