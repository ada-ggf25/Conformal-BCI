"""Tests for src/data/hybrid_io.py."""

from unittest.mock import MagicMock, patch

import numpy as np
from bson import ObjectId

from src.data.hybrid_io import (
    build_hybrid_neuron_mapping,
    create_windows_with_regions,
    extract_neural_and_position_hybrid,
    trial_water_position,
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
    @patch("src.data.hybrid_io.get_session_neuron_ids")
    def test_shared_ids_are_the_intersection(self, mock_ids):
        """The shared/specific split is the pooled model's input contract: a wrong
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
