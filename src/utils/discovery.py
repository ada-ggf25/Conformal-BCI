"""Session discovery and grouping utilities.

Per-mouse primitives over the ``subjects``, ``dates`` and ``trials`` collections;
``cohort.discover_cohort`` pools their results across mice.
"""

import re

from ..data.config import MIN_NEURONS_PER_SESSION
from ..data.hybrid_io import get_session_neuron_ids


def get_all_mice(db) -> list[int]:
    """Return all mouse IDs recorded in the subjects collection.

    Args:
        db (Database): pymongo Database.

    Returns:
        The ``mouse_id`` of every subject that has one, in collection order.
    """
    subjects = list(db["subjects"].find({}))
    return [s["mouse_id"] for s in subjects if "mouse_id" in s]


def get_contexts_for_mouse(db, mouse_id: int) -> dict[str, list[str]]:
    """Return {paradigm: [session_names]} for all sessions belonging to mouse_id.

    A session is included only if all of these hold:

    - the dates document has a matching ``pos_path`` (``M<id>_<YYMMDD>`` format);
    - at least one trial exists for that date;
    - at least one neural data file exists for that trial.

    Args:
        db (Database): pymongo Database.
        mouse_id:      the mouse's numeric id.

    Returns:
        A dict keyed by paradigm string (``"Unknown"`` when a date has none), with sorted,
        deduplicated session lists. Empty if the mouse is not in ``subjects``.
    """
    subject = db["subjects"].find_one({"mouse_id": mouse_id})
    if not subject:
        return {}

    mouse_db_id = subject["_id"]
    dates = list(db["dates"].find({"mouse_db_id": mouse_db_id}))

    contexts: dict[str, list[str]] = {}
    for date_doc in dates:
        paradigm = date_doc.get("paradigm", "Unknown")
        pos_path = date_doc.get("pos_path", "")
        date_id = date_doc["_id"]

        match = re.search(rf"M{mouse_id}_(\d{{6}})", pos_path)
        if not match:
            continue
        session_name = f"M{mouse_id}_{match.group(1)}"

        trial = db["trials"].find_one({"date_db_id": date_id})
        if not trial:
            continue
        nfile = db["neurons_data.files"].find_one(
            {"metadata.trial_db_id": trial["_id"]}
        )
        if not nfile:
            continue

        contexts.setdefault(paradigm, []).append(session_name)

    return {ctx: sorted(set(sessions)) for ctx, sessions in contexts.items()}


def resolve_paradigm(
    db,
    mouse_id: int,
    target_paradigm: str = "",
    min_neurons: int | None = None,
) -> tuple[str, list[str]]:
    """Pick the paradigm this mouse will be analyzed under, and its valid sessions.

    A session is valid when it has at least ``min_neurons`` curated units. A
    paradigm is eligible when it has at least two valid sessions.

    ``discover_cohort`` calls this once per mouse and pooled paradigm, to decide which
    of that mouse's sessions enter the pool.

    Args:
        db (Database):   pymongo Database.
        mouse_id:        the mouse's numeric id.
        target_paradigm: the paradigm to consider; empty considers every paradigm of
                         the mouse and picks the one with the most valid sessions.
        min_neurons:     minimum units for a valid session (None =
                         ``MIN_NEURONS_PER_SESSION``).

    Returns:
        (paradigm, valid_sessions): sessions sorted, length >= 2.

    Raises:
        RuntimeError: if the mouse has no sessions, or no eligible paradigm.
            ``discover_cohort`` catches it and records the mouse as skipped.
    """
    if min_neurons is None:
        min_neurons = MIN_NEURONS_PER_SESSION

    contexts = get_contexts_for_mouse(db, mouse_id)
    if not contexts:
        raise RuntimeError(f"No sessions found for Mouse {mouse_id} in {db.name}.")

    if target_paradigm:
        candidates = (
            {target_paradigm: contexts[target_paradigm]}
            if target_paradigm in contexts
            else {}
        )
    else:
        candidates = contexts

    group: list[str] = []
    paradigm = ""
    for name, sessions in candidates.items():
        if len(sessions) < 2:
            continue
        valid = [
            s
            for s in sorted(sessions)
            if len(get_session_neuron_ids(db, s)) >= min_neurons
        ]
        if len(valid) > len(group):
            group, paradigm = valid, name

    if len(group) < 2:
        raise RuntimeError(
            f"Mouse {mouse_id} has no paradigm with >=2 valid sessions "
            f"(>= {min_neurons} neurons)"
            + (f" for TARGET_PARADIGM={target_paradigm!r}." if target_paradigm else ".")
        )
    return paradigm, group
