"""Tests for src/data/regions.py.

Accepted gaps: none.
"""

from src.data.config import LINEAR_N_REGIONS
from src.data.regions import (
    classify_point_more_water,
    classify_point_nested,
    classify_point_ymaze,
    get_num_classes,
    uses_water,
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


def test_more_water_cuts_the_run_to_water_into_eight_equal_regions():
    """Region k covers [k/8, (k+1)/8) of the distance to the trial's water; the start and
    anything at or beyond the water clip to the first and last region."""
    water = 2500.0  # one region spans 312.5 units
    cases = [(-50, 0), (0, 0), (312, 0), (313, 1), (1250, 4), (2187, 6), (2188, 7),
             (2499, 7), (2500, 7), (5000, 7)]  # fmt: skip
    for y, want in cases:
        assert classify_point_more_water(y, water) == want, y


def test_more_water_regions_scale_with_the_water_position():
    """The same fraction of the run is the same region, whatever the track length."""
    for water in (2500.0, 7500.0, 27500.0):
        assert classify_point_more_water(0.55 * water, water) == 4


def test_more_water_without_a_water_position_is_unlabeled():
    """A trial with no usable water position yields -1, never a real region."""
    for water in (float("nan"), 0.0, -100.0):
        assert classify_point_more_water(1000.0, water) == -1


def test_class_count_covers_every_label_each_classifier_can_emit():
    """The decoder's output width comes from get_num_classes, the labels from the classifiers:
    every label a paradigm's classifier emits must be a valid class, and every class reachable.
    """
    grid = [(x, y) for x in range(-12000, 12001, 250) for y in range(-500, 12001, 250)]
    emitted = {
        "Y_Maze": {classify_point_ymaze(x, y) for x, y in grid},
        "Nested_Maze": {classify_point_nested(x, y) for x, y in grid},
        "More_Water": {
            classify_point_more_water(y, 2500.0) for y in range(-100, 2700, 10)
        },
    }
    for paradigm, labels in emitted.items():
        assert labels == set(range(get_num_classes(paradigm))), paradigm


def test_linear_region_count_is_the_configured_one():
    """The Linear maze's class count is the configured number of equal segments."""
    assert get_num_classes("More_Water") == LINEAR_N_REGIONS
    top = classify_point_more_water(10_000.0, 2500.0)
    assert top == LINEAR_N_REGIONS - 1


def test_only_the_linear_maze_uses_the_water_position():
    """Water-relative regions belong to the Linear maze; the Y and Nested mazes have fixed
    geometry, including the merged Nested_Maze+2 block."""
    for paradigm, expected in [
        ("More_Water", True),
        ("Y_Maze", False),
        ("Nested_Maze", False),
        ("Nested_Maze+2", False),
        ("Nested_All", False),
    ]:
        assert uses_water(paradigm) is expected, paradigm
