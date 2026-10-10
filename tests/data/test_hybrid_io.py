"""Tests for src/data/hybrid_io.py.

MongoDB is a MagicMock and the GridFS loaders are patched, so every trial's spike times and
behavior samples are chosen by hand and each expected bin is worked out from the geometry.

Accepted gaps: none.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from bson import ObjectId

from src.data.hybrid_io import (
    _classify_bin,
    _trial_bin_count,
    _xy_columns,
    build_example_raster,
    build_hybrid_neuron_mapping,
    create_windows_with_regions,
    extract_neural_and_position_hybrid,
    get_session_neuron_ids,
    trial_water_position,
    window_region_labels,
    window_task_mask,
)


def _mock_db():
    """Return a MagicMock standing in for the MongoDB handle."""
    return MagicMock()


def _trial(status, ticks):
    """Trial doc spanning `ticks` (ticks/1000 bins at 0.1 s)."""
    return {
        "_id": ObjectId(),
        "trial_start_neural_data": 0,
        "trial_end_neural_data": ticks,
        "trial_start_behavior_data": None,
        "trial_end_behavior_data": None,
        "trial_status": status,
    }


def _wire_trials(db, docs):
    """Wire the trials collection so find().sort() yields the given docs in order."""
    db["trials"].find.return_value.sort.return_value = [{"_id": d["_id"]} for d in docs]
    by_id = {d["_id"]: d for d in docs}
    db["trials"].find_one.side_effect = lambda q: by_id[q["_id"]]


class TestWindowTaskMask:
    """Only task windows enter the chronological split, so a wrong mask silently
    changes which data the model trains and calibrates on."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_task_and_nontask_trials_map_to_windows(self, beh, neu, did):
        """A task trial's window is masked True and a non-task trial's False."""
        db = _mock_db()
        did.return_value = ObjectId()
        neu.return_value = {"n1": [500]}
        beh.return_value = {"X_Position": [0.0], "Y_Position": [0.0]}
        # W trial = 10 bins = 1 window (all task), N trial = 10 bins = 1 (non-task).
        _wire_trials(db, [_trial("W", 10000), _trial("N", 10000)])
        assert window_task_mask(db, "M50_250411", window_size=10).tolist() == [
            True,
            False,
        ]


class TestCreateWindowsWithRegions:
    """A window's content and label must come from the same bins."""

    def test_region_assigned_from_last_bin_of_window(self):
        """A window takes its region label from its final bin, not its first or its mean."""
        neural = np.zeros((20, 2), dtype=np.float32)
        x, y = np.zeros(20), np.zeros(20)
        y[9] = 500  # window 0 last bin -> R0 (y < 1250)
        y[19], x[19] = 7000, 1.0  # window 1 last bin -> R7 (x>=0, y>=6000)
        _windows, regions = create_windows_with_regions(neural, x, y, 10, "Y_Maze")
        assert regions[0] == 0
        assert regions[1] == 7

    def test_more_water_paradigm_dispatch(self):
        """The other dispatch branch. Not implied by the Y_Maze test above, and not
        implied by window_region_labels agreeing with this function either."""
        _, regions = create_windows_with_regions(
            np.zeros((10, 2), dtype=np.float32),
            np.zeros(10),
            np.full(10, 1300.0),
            10,
            "More_Water",
            np.full(10, 2500.0),
        )
        assert regions[0] == 4  # 1300 / 2500 of the run -> region floor(8 * 0.52) = 4

    def test_window_content_matches_the_neural_slice(self):
        """A slicing bug would silently pair each window with another window's label."""
        rng = np.random.RandomState(1)
        neural = rng.rand(20, 3).astype(np.float32)
        windows, _ = create_windows_with_regions(
            neural, np.zeros(20), np.zeros(20), 10, "Y_Maze"
        )
        np.testing.assert_array_equal(windows[0], neural[0:10])
        np.testing.assert_array_equal(windows[1], neural[10:20])


class TestBuildHybridNeuronMapping:
    """Which neurons count as shared across the pooled sessions."""

    @patch("src.data.hybrid_io.get_session_neuron_ids")
    def test_shared_ids_are_the_intersection(self, mock_ids):
        """The shared/specific split is the encoder's input contract: a wrong
        partition routes a neuron through the wrong projection without any error."""
        mock_ids.side_effect = lambda db, name: {
            "A": ["n1", "n2", "n3"],
            "B": ["n2", "n3", "n4"],
        }[name]
        shared, _ = build_hybrid_neuron_mapping(_mock_db(), ["A", "B"])
        assert set(shared) == {"n2", "n3"}


_TRIAL_DOC = {
    "trial_start_neural_data": 0,
    "trial_end_neural_data": 10000,  # 1 second in 10000ths
    "trial_start_behavior_data": None,
    "trial_end_behavior_data": None,
}


def _one_trial_db(db):
    """Wire the trials collection to yield exactly one trial spanning one second."""
    trial_id = ObjectId()
    db["trials"].find.return_value.sort.return_value = [{"_id": trial_id}]
    db["trials"].find_one.return_value = dict(_TRIAL_DOC, _id=trial_id)


def _config(neuron_ids):
    """Return a minimal session_config mapping neuron ids to column indices."""
    return {
        "n_total": len(neuron_ids),
        "neuron_id_to_idx": {nid: i for i, nid in enumerate(neuron_ids)},
    }


class TestExtractNeuralAndPositionHybrid:
    """Spike times and positions become the model's input here, both expectations are
    worked by hand from the bin geometry."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_spike_appears_in_correct_bin(self, beh, neu, did):
        """Each spike lands in the bin covering its time, at a rate of count over bin width."""
        db = _mock_db()
        did.return_value = ObjectId()
        _one_trial_db(db)
        neu.return_value = {"n1": [500, 1500]}  # 0.05 s -> bin 0 ; 0.15 s -> bin 1
        beh.return_value = {"X_Position": [1.0] * 100, "Y_Position": [2.0] * 100}
        neural, _, _, _ = extract_neural_and_position_hybrid(
            db, "M21_250214", _config(["n1"]), 0.1
        )
        # Spike rate = count / bin_width = 1 / 0.1 = 10.0
        np.testing.assert_allclose(neural[0, 0], 10.0)
        np.testing.assert_allclose(neural[1, 0], 10.0)
        assert neural[2, 0] == 0.0  # bin 2 is empty

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_position_averaged_correctly_per_bin(self, beh, neu, did):
        """Each bin's position is the mean of the behavior samples falling inside it."""
        db = _mock_db()
        did.return_value = ObjectId()
        _one_trial_db(db)
        neu.return_value = {"n1": []}
        # 20 samples over 10 bins -> 2 per bin: [1,1,3,3,5,5,...]
        beh.return_value = {
            "X_Position": [float(2 * i + 1) for i in range(10) for _ in range(2)],
            "Y_Position": [0.0] * 20,
        }
        _, x_out, _, _ = extract_neural_and_position_hybrid(
            db, "M21_250214", _config(["n1"]), 0.1
        )
        np.testing.assert_allclose(x_out[0], 1.0)  # mean([1, 1])
        np.testing.assert_allclose(x_out[1], 3.0)  # mean([3, 3])


class TestTrialWaterPosition:
    """The Linear maze's regions are fractions of the run to this position, so a wrong value
    relabels every window of the trial."""

    def test_position_plus_distance_is_the_water(self):
        """Y + distance to water is constant within a trial and is the reward position."""
        y = np.linspace(0.0, 2400.0, 50)
        b = {
            "Y_Position": y.tolist(),
            "Direct_Distance_to_Water": (2500.0 - y).tolist(),
        }
        assert trial_water_position(b) == 2500.0

    def test_rounding_wobble_is_merged_into_the_mode(self):
        """Samples one 100-unit bin away (sideways offset in the distance) are the same water."""
        y = np.zeros(100)
        dist = np.r_[
            np.full(60, 7500.0), np.full(40, 7560.0)
        ]  # rounds to 7500 and 7600
        b = {"Y_Position": y.tolist(), "Direct_Distance_to_Water": dist.tolist()}
        assert trial_water_position(b) == 7500.0

    def test_ill_defined_water_is_rejected(self):
        """Below the minimum share of samples at the mode, the trial has no water position."""
        y = np.zeros(100)
        dist = np.r_[np.full(50, 2500.0), np.full(50, 7500.0)]
        b = {"Y_Position": y.tolist(), "Direct_Distance_to_Water": dist.tolist()}
        assert trial_water_position(b) is None

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_every_bin_of_a_trial_carries_its_water(self, beh, neu, did):
        """The extraction broadcasts the trial's water position to each of its bins."""
        db = _mock_db()
        did.return_value = ObjectId()
        _one_trial_db(db)
        neu.return_value = {"n1": []}
        beh.return_value = {
            "X_Position": [0.0] * 20,
            "Y_Position": [100.0] * 20,
            "Direct_Distance_to_Water": [2400.0] * 20,
        }
        *_, water = extract_neural_and_position_hybrid(
            db, "M21_250214", _config(["n1"]), 0.1
        )
        np.testing.assert_array_equal(water, np.full(10, 2500.0))


# --------------------------------------------------------------------------- neuron ids


class TestGetSessionNeuronIds:
    """A session's neurons, as the neuron-count gate sees them."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    def test_ids_are_the_first_trials_neural_keys(self, neu, did):
        """The session's neurons are the units present in its first trial's spike file."""
        db = _mock_db()
        did.return_value = ObjectId()
        tid = ObjectId()
        db["trials"].find_one.return_value = {"_id": tid}
        neu.return_value = {"a": [1.0], "b": []}
        assert get_session_neuron_ids(db, "M1_240101") == ["a", "b"]
        assert neu.call_args.args[1] == tid

    @patch("src.data.hybrid_io.session_name_to_date_id")
    def test_a_session_without_trials_has_no_neurons(self, did):
        """No trial means no neurons, which the MIN_NEURONS gate then drops."""
        db = _mock_db()
        did.return_value = ObjectId()
        db["trials"].find_one.return_value = None
        assert get_session_neuron_ids(db, "M1_240101") == []


class TestNeuronMappingIndices:
    """Every neuron's column and the projection it is read through."""

    @patch("src.data.hybrid_io.get_session_neuron_ids")
    def test_column_layout_of_each_session(self, mock_ids):
        """Columns follow each session's sorted ids; the shared and specific index lists
        partition them, so every neuron enters exactly one projection."""
        mock_ids.side_effect = lambda db, name: {
            "A": ["n3", "n1", "n2"],
            "B": ["n2", "n4", "n3"],
        }[name]
        shared, cfg = build_hybrid_neuron_mapping(_mock_db(), ["A", "B"])
        assert shared == ["n2", "n3"]
        a, b = cfg["A"], cfg["B"]
        assert a["all_ids"] == ["n1", "n2", "n3"]
        assert a["shared_indices"] == [1, 2] and a["specific_indices"] == [0]
        assert a["specific_ids"] == ["n1"]
        assert b["all_ids"] == ["n2", "n3", "n4"]
        assert b["shared_indices"] == [0, 1] and b["specific_indices"] == [2]
        assert (b["n_total"], b["n_shared"], b["n_specific"]) == (3, 2, 1)
        assert b["neuron_id_to_idx"] == {"n2": 0, "n3": 1, "n4": 2}


# --------------------------------------------------------------------------- trial geometry


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({"trial_start_neural_data": 0, "trial_end_neural_data": 10000}, 10),
        ({"trial_start_neural_data": 5000, "trial_end_neural_data": 17345}, 13),
        ({"trial_start_neural_data": None, "trial_end_neural_data": 10000}, None),
        ({"trial_start_neural_data": float("nan"), "trial_end_neural_data": 1}, None),
        ({"trial_start_neural_data": 10000, "trial_end_neural_data": 10000}, None),
        ({"trial_start_neural_data": 10000, "trial_end_neural_data": 500}, None),
        # no neural end: the behavior clock (days) supplies the duration, here 2 s
        (
            {
                "trial_start_neural_data": 0,
                "trial_end_neural_data": None,
                "trial_start_behavior_data": 100.0,
                "trial_end_behavior_data": 100.0 + 2 / 86400,
            },
            20,
        ),
        (
            {
                "trial_start_neural_data": 0,
                "trial_end_neural_data": float("nan"),
                "trial_start_behavior_data": 0.0,
                "trial_end_behavior_data": float("nan"),
            },
            None,
        ),
        ({"trial_start_neural_data": 0}, None),
    ],
)
def test_trial_bin_count(doc, expected):
    """A trial spans ceil(duration / 0.1 s) bins at 10 kHz ticks; unusable timing is None."""
    got = _trial_bin_count(doc, 0.1)
    assert got == expected


def test_xy_columns_with_the_plain_channel_names():
    """The unprefixed X_Position / Y_Position channels are found too."""
    assert _xy_columns({"Y_Position": [], "X_Position": []}) == (
        "X_Position",
        "Y_Position",
    )


@pytest.mark.parametrize(
    ("x", "y", "paradigm", "water", "expected"),
    [
        (1.0, 9000.0, "Y_Maze", np.nan, 7),
        (1.0, 9000.0, "Nested_Maze", np.nan, 7),
        (3000.0, 9000.0, "Nested_Maze+2", np.nan, 8),
        (1.0, 1300.0, "More_Water", 2600.0, 4),
        (1.0, 1300.0, "More_Water", np.nan, -1),
        (1.0, 1300.0, "Y_Maze", 2600.0, 1),  # the Y maze ignores the water
    ],
)
def test_classify_bin_dispatches_on_the_paradigm(x, y, paradigm, water, expected):
    """Each paradigm string routes the sample to its own maze geometry."""
    assert _classify_bin(x, y, paradigm, water) == expected


def test_xy_columns_are_found_by_substring():
    """Prefixed channel names are recognized; a document without them yields None."""
    assert _xy_columns(
        {"Mouse_X_Position": [], "Mouse_Y_Position": [], "Speed": []}
    ) == (
        "Mouse_X_Position",
        "Mouse_Y_Position",
    )
    assert _xy_columns({"Speed": []}) == (None, None)


class TestTrialWaterPositionEdges:
    """When a Linear trial has no usable water position."""

    def test_missing_channels_mean_no_water(self):
        """Without a Y channel or a distance channel the water position is undefined."""
        assert trial_water_position({"Direct_Distance_to_Water": [1.0]}) is None
        assert trial_water_position({"Y_Position": [1.0]}) is None

    def test_non_finite_samples_are_ignored(self):
        """NaN samples (tracking dropouts) do not count against the modal share."""
        y = [0.0, np.nan, 100.0, np.nan]
        d = [2500.0, 1.0, 2400.0, np.nan]
        assert (
            trial_water_position({"Y_Position": y, "Direct_Distance_to_Water": d})
            == 2500.0
        )

    def test_all_samples_missing_means_no_water(self):
        """A trial whose every sample is NaN has no water position."""
        b = {"Y_Position": [np.nan] * 3, "Direct_Distance_to_Water": [1.0] * 3}
        assert trial_water_position(b) is None


# --------------------------------------------------------------------------- extraction


def _wire_trial_data(db, docs, neural, behavior):
    """Wire trials, and per-trial neural / behavior payloads keyed by trial id."""
    _wire_trials(db, docs)
    return (
        lambda _db, tid: neural.get(tid, {}),
        lambda _db, tid: behavior.get(tid, {}),
    )


class TestExtractionAcrossTrials:
    """Which trials extraction keeps, and how their bins line up."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_unusable_trials_are_skipped_and_the_rest_concatenate(self, beh, neu, did):
        """Of six trials only two are usable; their bins are stacked in trial order, and
        spike times are relative to each trial's own start (bin 0 of every trial)."""
        db = _mock_db()
        did.return_value = ObjectId()
        good1, no_time, no_neural, no_behavior, no_xy, good2 = (
            _trial("W", 10000),
            dict(_trial("W", 0), trial_end_neural_data=0),
            _trial("W", 10000),
            _trial("W", 10000),
            _trial("W", 10000),
            _trial("W", 20000),
        )
        xy = {"X_Position": [0.0] * 10, "Y_Position": [1.0] * 10}
        n_fn, b_fn = _wire_trial_data(
            db,
            [good1, no_time, no_neural, no_behavior, no_xy, good2],
            {
                good1["_id"]: {"a": [500], "ghost": [500]},
                no_time["_id"]: {"a": [1]},
                no_behavior["_id"]: {"a": [1]},
                no_xy["_id"]: {"a": [1]},
                good2["_id"]: {"a": [500, 600], "b": [15500]},
            },
            {
                good1["_id"]: xy,
                no_time["_id"]: xy,
                no_neural["_id"]: xy,
                no_xy["_id"]: {"Speed": [1.0]},
                good2["_id"]: {"X_Position": [2.0] * 20, "Y_Position": [3.0] * 20},
            },
        )
        neu.side_effect, beh.side_effect = n_fn, b_fn
        neural, x, y, water = extract_neural_and_position_hybrid(
            db, "M1_240101", _config(["a", "b"]), 0.1
        )
        assert neural.shape == (30, 2)  # 10 bins + 20 bins
        assert neural[0, 0] == 10.0 and neural[10, 0] == 20.0  # 1 and 2 spikes / 0.1 s
        assert (
            neural[25, 1] == 10.0
        )  # b's spike at 1.55 s -> bin 15 of trial 2 = row 25
        assert neural.sum() == 40.0  # the unmapped "ghost" unit contributes nothing
        np.testing.assert_array_equal(x, np.r_[np.zeros(10), np.full(20, 2.0)])
        np.testing.assert_array_equal(y, np.r_[np.ones(10), np.full(20, 3.0)])
        assert np.isnan(water).all()  # no distance channel: no water position

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_no_usable_trial_yields_nones(self, beh, neu, did):
        """A session none of whose trials is usable returns four Nones, which callers skip."""
        db = _mock_db()
        did.return_value = ObjectId()
        _wire_trials(db, [_trial("W", 10000)])
        neu.return_value = {}
        assert extract_neural_and_position_hybrid(
            db, "M1_240101", _config(["a"]), 0.1
        ) == (None, None, None, None)

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_sparse_behavior_leaves_uncovered_bins_at_zero(self, beh, neu, did):
        """With fewer behavior samples than bins, a bin that no sample falls into stays 0."""
        db = _mock_db()
        did.return_value = ObjectId()
        _one_trial_db(db)  # 10 bins
        neu.return_value = {"n1": []}
        beh.return_value = {"X_Position": [4.0] * 5, "Y_Position": [8.0] * 5}
        _, x, y, _ = extract_neural_and_position_hybrid(
            db, "M1_240101", _config(["n1"]), 0.1
        )
        # 0.5 samples per bin: bin i covers samples [int(i/2), int((i+1)/2)), which is empty
        # for even i ([0, 0), [1, 1), ...) and one sample for odd i ([0, 1), [1, 2), ...)
        np.testing.assert_array_equal(x, [0.0, 4.0] * 5)
        np.testing.assert_array_equal(y, [0.0, 8.0] * 5)


class TestWindowTaskMaskEdges:
    """The task mask must line up window for window with extraction."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_window_straddling_a_non_task_trial_is_not_task(self, beh, neu, did):
        """A window counts as task only if all of its bins are; the partial tail is dropped.

        W trial 15 bins + N trial 10 bins = 25 bins -> windows [0,10) all W, [10,20) mixed.
        """
        db = _mock_db()
        did.return_value = ObjectId()
        neu.return_value = {"n1": [1]}
        beh.return_value = {"X_Position": [0.0], "Y_Position": [0.0]}
        _wire_trials(db, [_trial("W", 15000), _trial("N", 10000)])
        assert window_task_mask(db, "M1_240101", 10).tolist() == [True, False]

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    @patch("src.data.hybrid_io.get_behavior")
    def test_mask_skips_exactly_the_trials_extraction_skips(self, beh, neu, did):
        """The mask must line up with the extracted windows, so a trial the extraction drops
        (no timing, no spikes, no behavior, no XY) must contribute no bins here either."""
        db = _mock_db()
        did.return_value = ObjectId()
        docs = [
            dict(_trial("N", 0), trial_start_neural_data=None),
            _trial("N", 10000),  # no neural
            _trial("N", 10000),  # no behavior
            _trial("N", 10000),  # no XY
            _trial("W", 10000),
        ]
        n_fn, b_fn = _wire_trial_data(
            db,
            docs,
            {d["_id"]: {"n": [1]} for d in docs[2:]},
            {
                docs[3]["_id"]: {"Speed": [0.0]},
                docs[4]["_id"]: {"X_Position": [0.0], "Y_Position": [0.0]},
            },
        )
        neu.side_effect, beh.side_effect = n_fn, b_fn
        assert window_task_mask(db, "M1_240101", 10).tolist() == [True]

    @patch("src.data.hybrid_io.session_name_to_date_id")
    def test_no_usable_trial_gives_an_empty_mask(self, did):
        """No usable trial means zero windows."""
        db = _mock_db()
        did.return_value = ObjectId()
        _wire_trials(db, [])
        mask = window_task_mask(db, "M1_240101", 10)
        assert mask.dtype == bool and len(mask) == 0


# --------------------------------------------------------------------------- raster


class TestBuildExampleRaster:
    """The illustrative raster reads the first trial that has spikes."""

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    def test_rates_are_counted_from_the_first_spike_of_the_first_spiking_trial(
        self, neu, did
    ):
        """The first trial has no spikes, so the second is used. Times are shifted to its
        first spike (1.0 s); n1's spikes at 0, 0.05 and 1.5 s land in 0.5 s bins 0, 0, 3,
        its spike at 2.5 s falls outside the 2 s window."""
        db = _mock_db()
        did.return_value = ObjectId()
        empty, spiking = ObjectId(), ObjectId()
        db["trials"].find.return_value.sort.return_value = [
            {"_id": empty},
            {"_id": spiking},
        ]
        neu.side_effect = lambda _db, tid: (
            {}
            if tid == empty
            else {"n2": [12000], "n1": [10000, 10500, 25000, 35000], "silent": []}
        )
        out = build_example_raster(
            db, "M1_240101", window_sec=2.0, max_neurons=2, bin_width_sec=0.5
        )
        assert out["n_shown"] == 2
        np.testing.assert_allclose(
            out["rates"][0], [4.0, 0.0, 0.0, 2.0]
        )  # n1: most spikes
        np.testing.assert_allclose(out["rates"][1], [2.0, 0.0, 0.0, 0.0])  # n2 at 0.2 s

    @patch("src.data.hybrid_io.session_name_to_date_id")
    @patch("src.data.hybrid_io.get_neural")
    def test_a_session_without_spikes_has_no_raster(self, neu, did):
        """If no trial has spike data there is nothing to draw."""
        db = _mock_db()
        did.return_value = ObjectId()
        db["trials"].find.return_value.sort.return_value = [{"_id": ObjectId()}]
        neu.return_value = {}
        assert build_example_raster(db, "M1_240101") is None


# --------------------------------------------------------------------------- windows


class TestWindowing:
    """Windowing edge cases: the last-bin label, the tail, and dispatch."""

    def test_label_reads_x_from_the_last_bin_too(self):
        """At y = 7000 the Y maze splits on the sign of x: only the last bin is on the right,
        so the window is R7, not R6."""
        x = np.r_[np.full(9, -5.0), 5.0]
        _, r = create_windows_with_regions(
            np.zeros((10, 1), np.float32), x, np.full(10, 7000.0), 10, "Y_Maze"
        )
        assert r.tolist() == [7]

    def test_trailing_partial_window_is_dropped(self):
        """25 bins at 10 per window is 2 windows; the last 5 bins belong to none."""
        w, r = create_windows_with_regions(
            np.zeros((25, 1), np.float32), np.zeros(25), np.zeros(25), 10, "Y_Maze"
        )
        assert len(w) == len(r) == 2

    def test_nested_paradigm_dispatch(self):
        """A Nested paradigm is labeled by the nested geometry, not the Y maze's."""
        x, y = np.full(10, 3000.0), np.full(10, 9000.0)  # nested R8, Y-maze R7
        _, r = create_windows_with_regions(
            np.zeros((10, 1), np.float32), x, y, 10, "Nested_Maze+2"
        )
        assert r.tolist() == [8]

    def test_linear_windows_without_water_are_unlabeled(self):
        """With no water array a Linear window gets -1, which the task mask then excludes."""
        _, r = create_windows_with_regions(
            np.zeros((10, 1), np.float32),
            np.zeros(10),
            np.full(10, 500.0),
            10,
            "More_Water",
        )
        assert r.tolist() == [-1]


class TestWindowRegionLabels:
    # Y maze, 3 windows of 4 bins. Bin labels by y: <1250 -> 0, <2500 -> 1, x>=0 & <4250 -> 4.
    """The last and mode reductions, and the straddle flag."""

    Y = np.array(
        [0, 0, 0, 0, 0, 1300, 1300, 1300, 1300, 1300, 3000, 0], dtype=float
    )  # bin labels: (0, 0, 0, 0), (0, 1, 1, 1), (1, 1, 4, 0)
    X = np.zeros(12)

    def test_last_reduction_matches_create_windows(self):
        """reduce="last" is the same labeling the window extraction applies."""
        regions, _ = window_region_labels(self.X, self.Y, 4, "Y_Maze", reduce="last")
        _, expected = create_windows_with_regions(
            np.zeros((12, 1), np.float32), self.X, self.Y, 4, "Y_Maze"
        )
        assert regions.tolist() == expected.tolist() == [0, 1, 0]

    def test_mode_reduction_takes_the_most_frequent_bin_label(self):
        """Window 2 ends in R0 but spends two of its four bins in R1: the mode is R1."""
        regions, _ = window_region_labels(self.X, self.Y, 4, "Y_Maze", reduce="mode")
        assert regions.tolist() == [0, 1, 1]

    def test_straddle_flags_windows_crossing_a_boundary(self):
        """Only windows whose bins span more than one region are straddlers."""
        _, straddle = window_region_labels(self.X, self.Y, 4, "Y_Maze")
        assert straddle.tolist() == [False, True, True]

    def test_linear_labels_use_each_bins_water(self):
        """Linear bins are labeled against their own trial's water, bin by bin."""
        y = np.array([1000.0, 1000.0])
        water = np.array([2000.0, 8000.0])  # halfway -> R4 ; an eighth -> R1
        regions, straddle = window_region_labels(
            np.zeros(2), y, 1, "More_Water", water=water
        )
        assert regions.tolist() == [4, 1]
        assert not straddle.any()
