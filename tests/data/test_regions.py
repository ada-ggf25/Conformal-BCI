"""Tests for src/data/regions.py."""

from src.data.regions import (
    classify_point_more_water,
    classify_point_nested,
    classify_point_ymaze,
)


def test_ymaze_each_region_maps_to_its_zone():
    """Representative points on every Y-maze boundary map to their region (8, R0-R7)."""
    cases = [
        ((0, 0), 0), ((-5000, 1249), 0), ((5000, 1249), 0),
        ((0, 1250), 1), ((-5000, 2499), 1), ((5000, 2000), 1),
        ((-1, 2500), 2), ((-1000, 4249), 2),
        ((-1, 4250), 3), ((-1000, 5999), 3),
        ((0, 2500), 4), ((1000, 4249), 4),
        ((0, 4250), 5), ((1000, 5999), 5),
        ((-1, 6000), 6), ((-5000, 10000), 6),
        ((0, 6000), 7), ((5000, 10000), 7),
    ]  # fmt: skip
    for (x, y), want in cases:
        assert classify_point_ymaze(x, y) == want, (x, y)


def test_nested_each_region_maps_to_its_zone():
    """Representative points on every nested boundary map to their region (9, R0-R8)."""
    cases = [
        ((0, 0), 0), ((-5000, 2499), 0), ((5000, 2499), 0),
        ((-1, 2500), 1), ((-1000, 5999), 1),
        ((0, 2500), 2), ((1000, 5999), 2),
        ((-1, 6000), 3), ((-1000, 8499), 3),
        ((0, 6000), 4), ((1000, 8499), 4),
        ((-2501, 8500), 5), ((-10000, 10000), 5),
        ((-2500, 8500), 6), ((-1, 9000), 6),
        ((0, 8500), 7), ((2499, 9000), 7),
        ((2500, 8500), 8), ((10000, 10000), 8),
    ]  # fmt: skip
    for (x, y), want in cases:
        assert classify_point_nested(x, y) == want, (x, y)


def test_more_water_bins_along_y_with_half_open_edges():
    """Each y value falls in the bin whose lower edge it meets; n regions from a list."""
    bounds = [0, 100, 200, 300]  # -> 3 regions
    for y, want in [(0, 0), (50, 0), (100, 1), (150, 1), (250, 2)]:
        assert classify_point_more_water(y, bounds) == want, y
