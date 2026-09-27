"""Spatial region classifiers for the Functional Atlas maze paradigms.

Maps a single (x, y) position reading to an integer region label R0-R7
(or R0-R8 for the Nested paradigm), called per time-bin during windowing
to assign region labels to neural windows.
"""

from __future__ import annotations

import math

from .config import LINEAR_N_REGIONS


def classify_point_ymaze(x: float, y: float) -> int:
    """Y-Maze: 8 regions (R0-R7) based on (x, y) position.

    The stem is cut at ``y`` = 1250 (R0) and 2500 (R1). Above that the maze forks by the
    sign of ``x``: the left arm is R2, R3 and R6 and the right arm is R4, R5 and R7, cut at
    ``y`` = 4250 and 6000.

    Args:
        x: lateral position, negative on the left arm.
        y: position along the maze from the start.

    Returns:
        The region label, 0-7.
    """
    if y < 1250:
        return 0
    if y < 2500:
        return 1
    if x < 0:
        if y < 4250:
            return 2
        if y < 6000:
            return 3
        return 6
    if y < 4250:
        return 4
    if y < 6000:
        return 5
    return 7


def classify_point_nested(x: float, y: float) -> int:
    """Nested maze: 9 regions (R0-R8) based on (x, y) position.

    The stem (``y`` < 2500) is R0. It splits by the sign of ``x`` into R1/R2 (up to
    ``y`` = 6000) and R3/R4 (up to 8500). Beyond that each side splits again at
    ``|x|`` = 2500: R5/R6 on the left, R7/R8 on the right.

    Args:
        x: lateral position, negative on the left side.
        y: position along the maze from the start.

    Returns:
        The region label, 0-8.
    """
    if y < 2500:
        return 0
    if y < 6000:
        return 1 if x < 0 else 2
    if y < 8500:
        return 3 if x < 0 else 4
    if x < 0:
        return 5 if x < -2500 else 6
    return 7 if x < 2500 else 8


def classify_point_more_water(
    y: float, water: float, n_regions: int = LINEAR_N_REGIONS
) -> int:
    """Linear maze: ``n_regions`` equal segments of the run from the start (y = 0) to the water.

    ``water`` is the trial's reward position. A position at or beyond it falls in the last
    region, and a position behind the start falls in the first.

    Args:
        y:         position along the track.
        water:     the trial's water position (``trial_water_position``).
        n_regions: number of equal segments.

    Returns:
        The region label in ``[0, n_regions)``, or -1 when the trial has no usable water
        position (NaN or not positive); such windows are kept out of the task windows.
    """
    if not math.isfinite(water) or water <= 0:
        return -1
    return int(min(n_regions - 1, max(0, math.floor(n_regions * y / water))))


def uses_water(paradigm: str) -> bool:
    """True for the Linear maze, whose regions are defined relative to each trial's water.

    Any paradigm that is neither a Y-Maze nor a Nested maze is treated as Linear, the same
    fallback ``get_num_classes`` and the region dispatcher use.

    Args:
        paradigm: paradigm string, e.g. ``"More_Water"``.

    Returns:
        Whether region labels need each trial's water position.
    """
    return "Y_Maze" not in paradigm and "Nested" not in paradigm


def get_num_classes(paradigm: str) -> int:
    """Return the number of region classes for a given paradigm string.

    Args:
        paradigm: paradigm string or cohort label; matched by substring, so
            ``"Nested_Maze+2"`` and ``"Nested_All"`` both count as Nested.

    Returns:
        9 for Nested, otherwise 8 (Y-Maze and Linear, ``LINEAR_N_REGIONS``).
    """
    if "Y_Maze" in paradigm:
        return 8
    if "Nested" in paradigm:
        return 9
    return 8
