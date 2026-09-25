"""Spatial region classifiers for the Functional Atlas maze paradigms.

Maps a single (x, y) position reading to an integer region label R0-R7
(or R0-R8 for the Nested paradigm), called per time-bin during windowing
to assign region labels to neural windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def classify_point_ymaze(x: float, y: float) -> int:
    """Y-Maze: 8 regions (R0-R7) based on (x, y) position."""
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
    """Nested maze: 9 regions (R0-R8) based on (x, y) position."""
    if y < 2500:
        return 0
    if y < 6000:
        return 1 if x < 0 else 2
    if y < 8500:
        return 3 if x < 0 else 4
    if x < 0:
        return 5 if x < -2500 else 6
    return 7 if x < 2500 else 8


def classify_point_more_water(y: float, y_boundaries: np.ndarray | list) -> int:
    """More-Water linear maze: n regions along the y-axis."""
    for i in range(len(y_boundaries) - 1):
        if y < y_boundaries[i + 1]:
            return i
    return len(y_boundaries) - 2


def get_num_classes(paradigm: str) -> int:
    """Return the number of region classes for a given paradigm string."""
    if "Y_Maze" in paradigm:
        return 8
    if "Nested" in paradigm:
        return 9
    return 8
