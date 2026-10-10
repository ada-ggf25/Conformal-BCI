"""Pooled-cohort discovery for the per-maze pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..data.regions import get_num_classes
from .discovery import get_all_mice, resolve_paradigm

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymongo.database import Database


MERGED: dict[str, list[str]] = {
    "Nested_All": ["Nested_Maze", "Nested_Maze+2"],
}
"""Named presets that pool several paradigm blocks into one cohort.

The three analyzed mazes are ``More_Water``, ``Y_Maze`` and ``Nested_All``. ``Nested_All``
is the only multi-block preset among them.
"""


def maze_family(paradigm: str) -> str:
    """Coarse maze family a paradigm string belongs to.

    The family decides the region geometry, so paradigms of one family can be pooled.

    Args:
        paradigm: paradigm string or cohort label.

    Returns:
        ``"Y_Maze"``, ``"Nested"`` or ``"More_Water"`` (the Linear maze, and the fallback).
    """
    if "Y_Maze" in paradigm:
        return "Y_Maze"
    return "Nested" if "Nested" in paradigm else "More_Water"


def assert_single_family(paradigms: list[str], label: str) -> None:
    """Raise unless every pooled paradigm and the display label share one family.

    The label is checked too, not just the members: a caller-supplied
    ``paradigm_label`` is free text, and a mislabeled merge would route the
    region labeler to the wrong geometry.

    Args:
        paradigms: the pooled paradigm strings.
        label:     the cohort's display label.

    Raises:
        AssertionError: if the paradigms and the label span more than one family.
    """
    families = {maze_family(p) for p in [*paradigms, label]}
    assert len(families) == 1, (
        f"cannot pool across maze families: {paradigms} labeled {label!r} spans {sorted(families)}"
    )


@dataclass
class CohortResult:
    """The pooled cohort: which paradigms, mice, and sessions survived discovery.

    Attributes:
        paradigms:           the pooled paradigm strings.
        label:               display and artifact label of the cohort.
        n_classes:           number of regions of the cohort's maze family.
        group:               every pooled session, sorted by name.
        session_to_mouse:    session name to mouse id.
        session_to_paradigm: session name to the paradigm it was pooled under.
        cohort_mice:         mice that contributed at least one session, in pool order.
        skipped:             ``(mouse_id, reason)`` for every mouse that contributed none.
    """

    paradigms: list[str]
    label: str
    n_classes: int
    group: list[str]
    session_to_mouse: dict[str, int]
    session_to_paradigm: dict[str, str]
    cohort_mice: list[int]
    skipped: list[tuple[int, str]]


def discover_cohort(
    db: Database,
    target_paradigm: str | list[str],
    *,
    target_mice: list[int] | None = None,
    exclude_mice: list[int] | tuple[int, ...] = (),
    paradigm_label: str = "",
) -> CohortResult:
    """Pool every usable mouse in ``target_paradigm`` into one cohort.

    Args:
        db:             pymongo Database.
        target_paradigm: a single paradigm string, a merged-preset name (a key of
                        ``MERGED``), or a list of paradigm strings to pool.
        target_mice:    restrict the pool to these mouse ids. ``None`` = every
                        mouse from ``get_all_mice``.
        exclude_mice:   mouse ids to hold out of the pool.
        paradigm_label: display / artifact label, replacing the default (the paradigm
                        itself, the preset name, or a list joined with "+"). It must
                        name the same maze family as the pooled paradigms.

    Returns:
        A ``CohortResult``. A mouse contributes the valid sessions of every pooled
        paradigm that ``resolve_paradigm`` accepts; one with none is listed in
        ``skipped``.

    Raises:
        RuntimeError: if no mouse has >= 2 valid sessions in any pooled paradigm.
        AssertionError: if the pool spans maze families or produces a duplicate
            session name.
    """
    if isinstance(target_paradigm, str) and target_paradigm in MERGED:
        # a merged-preset name (e.g. "Nested_All")
        paradigms = list(MERGED[target_paradigm])
        label = paradigm_label or target_paradigm
    elif isinstance(target_paradigm, str):
        paradigms = [target_paradigm]
        label = paradigm_label or target_paradigm
    else:
        paradigms = list(target_paradigm)
        label = paradigm_label or (
            paradigms[0] if len(paradigms) == 1 else "+".join(paradigms)
        )
    assert_single_family(paradigms, label)
    n_classes = get_num_classes(label)

    all_mice = sorted(get_all_mice(db)) if target_mice is None else list(target_mice)
    all_mice = [m for m in all_mice if m not in set(exclude_mice)]

    group: list[str] = []
    session_to_mouse: dict[str, int] = {}
    session_to_paradigm: dict[str, str] = {}
    cohort_mice: list[int] = []
    skipped: list[tuple[int, str]] = []

    for mid in all_mice:
        whys: list[str] = []
        contributed = False
        for par in paradigms:
            try:
                _pg, sess = resolve_paradigm(db, mid, par)
            except RuntimeError as exc:  # < 2 valid sessions / no such paradigm -> skip
                whys.append(f"{par}: {str(exc).splitlines()[0]}")
                continue
            for s in sess:
                group.append(s)
                session_to_mouse[s] = mid
                session_to_paradigm[s] = par
                contributed = True
        if contributed:
            cohort_mice.append(mid)
        else:
            skipped.append((mid, "; ".join(whys)))

    # session names are globally unique (M<id>_<YYMMDD>), so pooling never collides across mice
    assert len(group) == len(set(group)), (
        "duplicate session name across mice - pooling would double-count"
    )
    group = sorted(group)
    if not cohort_mice:
        raise RuntimeError(
            f"No usable mice for target_paradigm={target_paradigm!r}: none has >=2 valid sessions."
        )
    return CohortResult(
        paradigms=paradigms,
        label=label,
        n_classes=n_classes,
        group=group,
        session_to_mouse=session_to_mouse,
        session_to_paradigm=session_to_paradigm,
        cohort_mice=cohort_mice,
        skipped=skipped,
    )


def apply_session_exclusions(
    sessions: list[str],
    exclude: list[str] | None,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """Drop hand-picked sessions from the candidate pool before assembly.

    Args:
        sessions: candidate session ids.
        exclude:  session ids to drop. ``None`` or empty keeps every session.
        log_fn:   optional sink for a one-line summary of what was dropped (pass
                  ``print``); requested sessions not in the pool are reported too.

    Returns:
        The kept session ids, in their original order.
    """
    drop = set(exclude or ())
    unknown = drop - set(sessions)
    if unknown and log_fn is not None:
        log_fn(f"  EXCLUDE_SESSIONS: {sorted(unknown)} not in the cohort, ignored")
    kept = [s for s in sessions if s not in drop]
    if log_fn is not None and len(kept) != len(sessions):
        dropped = [s for s in sessions if s in drop]
        log_fn(
            f"  EXCLUDE_SESSIONS: dropped {len(dropped)} session(s) before assembly: {dropped}"
        )
    return kept
