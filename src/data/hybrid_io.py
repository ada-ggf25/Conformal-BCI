"""Neural data extraction and windowing pipeline.

Turns a session's trials in MongoDB into the arrays the models consume: one column per
neuron (``build_hybrid_neuron_mapping``), spike rates binned at ``BIN_WIDTH_SEC`` with
position averaged onto the same bins (``extract_neural_and_position_hybrid``), then
non-overlapping windows labeled with a maze region (``create_windows_with_regions``,
``window_region_labels``) and flagged as task or not (``window_task_mask``).

Timestamps: neural spike and trial times are sample indices at 10 kHz; behavior trial
bounds are in days.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymongo.database import Database

from ..db.database_utils import session_name_to_date_id
from ..db.loader import get_behavior, get_neural
from .config import BIN_WIDTH_SEC, LINEAR_WATER_MIN_SHARE
from .regions import (
    classify_point_more_water,
    classify_point_nested,
    classify_point_ymaze,
)


def get_session_neuron_ids(db, session_name: str) -> list[str]:
    """Return all neuron ID strings recorded for the first trial of a session.

    Only the first trial is read: its units are taken as the session's units.

    Args:
        db (Database): pymongo Database.
        session_name: e.g. ``"M21_250214"``.

    Returns:
        The neuron ids (``str(neurons._id)``), or an empty list if the session has no trial.
    """
    date_id = session_name_to_date_id(db, session_name)
    trial = db["trials"].find_one({"date_db_id": date_id})
    if not trial:
        return []
    ndata = get_neural(db, trial["_id"])
    return list(ndata.keys())


def build_example_raster(
    db: Database,
    session_name: str,
    *,
    window_sec: float = 30.0,
    max_neurons: int = 40,
    bin_width_sec: float = BIN_WIDTH_SEC,
    samples_per_sec: float = 10_000,
) -> dict | None:
    """Firing-rate raster over the first ``window_sec`` seconds of a session's first spiking trial.

    Illustrative only, built for the cross-maze notebook's single example figure, not part of
    the training pipeline.

    Args:
        db:              pymongo Database.
        session_name:    e.g. ``"M19_240918"``.
        window_sec:      length of the window shown, from the trial's first spike (default 30 s).
        max_neurons:     cap on rows in the raster, highest-firing neurons first (default 40).
        bin_width_sec:   time bin width in seconds (default: the pipeline's own BIN_WIDTH_SEC).
        samples_per_sec: spike sample-index rate (10 kHz).

    Returns:
        ``{"rates": (n_shown, n_bins) array of Hz, "n_shown": int}``, or ``None`` if no trial in
        the session has any spike data.
    """
    date_id = session_name_to_date_id(db, session_name)
    trial_ids = [
        t["_id"]
        for t in db["trials"]
        .find({"date_db_id": date_id})
        .sort("trial_start_neural_data", 1)
    ]
    ndata: dict = {}
    for tid in trial_ids:  # first trial WITH spike data, not just the first trial
        ndata = get_neural(db, tid)
        if ndata:
            break
    if not ndata:
        return None

    spikes_sec = {
        nid: np.asarray(t, dtype=float) / samples_per_sec for nid, t in ndata.items()
    }
    t0 = min(arr.min() for arr in spikes_sec.values() if arr.size)
    windowed = {
        nid: arr[(arr >= t0) & (arr < t0 + window_sec)] - t0
        for nid, arr in spikes_sec.items()
    }
    shown = sorted(windowed, key=lambda nid: -len(windowed[nid]))[:max_neurons]

    n_bins = int(np.ceil(window_sec / bin_width_sec))
    edges = np.arange(n_bins + 1) * bin_width_sec
    rates = np.array(
        [np.histogram(windowed[nid], bins=edges)[0] / bin_width_sec for nid in shown]
    )
    return {"rates": rates, "n_shown": len(shown)}


def build_hybrid_neuron_mapping(db, session_names: list[str]) -> tuple[list[str], dict]:
    """Compute the shared neuron set and per-session column index configs.

    A session's columns are its own neuron ids in sorted order. A neuron recorded in every
    session is "shared", and every other neuron is "specific" to its session. The encoder
    reads the two groups through separate projections.

    Args:
        db (Database): pymongo Database.
        session_names: the sessions to map; each must have at least one trial.

    Returns:
        ``(shared_neuron_ids, session_configs)``. ``shared_neuron_ids`` is the sorted list of
        neuron ids present in every session. ``session_configs`` maps each session name to a
        dict with keys:

        - ``all_ids``: the session's neuron ids, sorted; column ``i`` is ``all_ids[i]``.
        - ``specific_ids``: the ids not shared with every other session.
        - ``n_total``, ``n_shared``, ``n_specific``: the column counts.
        - ``shared_indices``: the shared neurons' columns, in ``shared_neuron_ids`` order.
        - ``specific_indices``: the specific neurons' columns, ascending.
        - ``neuron_id_to_idx``: neuron id to column.
    """
    session_neuron_ids: dict[str, set] = {}
    for session_name in session_names:
        neuron_ids = get_session_neuron_ids(db, session_name)
        session_neuron_ids[session_name] = set(neuron_ids)

    shared_neuron_ids = sorted(list(set.intersection(*session_neuron_ids.values())))
    shared_set = set(shared_neuron_ids)

    session_configs: dict = {}
    for session_name in session_names:
        all_ids = sorted(list(session_neuron_ids[session_name]))
        specific_ids = [nid for nid in all_ids if nid not in shared_set]

        shared_indices = [
            all_ids.index(nid)
            for nid in shared_neuron_ids
            if nid in session_neuron_ids[session_name]
        ]
        specific_indices = [i for i, nid in enumerate(all_ids) if nid not in shared_set]

        session_configs[session_name] = {
            "all_ids": all_ids,
            "specific_ids": specific_ids,
            "n_total": len(all_ids),
            "n_shared": len(shared_indices),
            "n_specific": len(specific_indices),
            "shared_indices": shared_indices,
            "specific_indices": specific_indices,
            "neuron_id_to_idx": {nid: i for i, nid in enumerate(all_ids)},
        }

    return shared_neuron_ids, session_configs


def _trial_bin_count(doc: dict, bin_width_sec: float) -> int | None:
    """Number of neural bins a trial document spans, or None if it is unusable.

    The trial's neural end time falls back to its start plus the behavior duration when
    it is missing. Behavior times are in days and neural times are 10 kHz sample indices,
    hence the ``86400 * 10000`` conversion.

    Args:
        doc:           a ``trials`` document.
        bin_width_sec: time bin width in seconds.

    Returns:
        ``ceil(duration / bin_width_sec)``, or None when the start is missing, both end
        times are missing, or the duration is not positive.
    """
    t_start_n = doc.get("trial_start_neural_data")
    t_end_n = doc.get("trial_end_neural_data")
    t_start_b = doc.get("trial_start_behavior_data")
    t_end_b = doc.get("trial_end_behavior_data")

    if t_start_n is None or np.isnan(t_start_n):
        return None
    if t_end_n is None or np.isnan(t_end_n):
        if t_end_b is None or np.isnan(t_end_b):
            return None
        t_end_n = t_start_n + (t_end_b - t_start_b) * 86400 * 10000

    duration = (t_end_n - t_start_n) / 10000
    if duration <= 0:
        return None
    return int(np.ceil(duration / bin_width_sec))


def _xy_columns(bdata: dict) -> tuple[str | None, str | None]:
    """Names of the X and Y position channels in a behavior document.

    Channel names vary between recordings (a prefix or suffix around ``X_Position``), so
    any key containing the name matches; the last match wins.

    Args:
        bdata: a behavior document from ``get_behavior``.

    Returns:
        ``(x_col, y_col)``, each None if no such channel exists.
    """
    x_col = y_col = None
    for key in bdata:
        if "X_Position" in key or key == "X_Position":
            x_col = key
        elif "Y_Position" in key or key == "Y_Position":
            y_col = key
    return x_col, y_col


WATER_CHANNEL = "Direct_Distance_to_Water"
"""Behavior channel with the distance from the animal to the trial's water reward."""


def trial_water_position(
    bdata: dict, min_share: float = LINEAR_WATER_MIN_SHARE
) -> float | None:
    """The trial's water reward position along the track, or None if it is ill-defined.

    ``Y_Position + Direct_Distance_to_Water`` is the reward's position and is constant within a
    trial. Samples are rounded to 100 units; the modal value, merged with its +-100 neighbors
    (rounding wobble), must cover at least ``min_share`` of the samples.

    Args:
        bdata:     a behavior document from ``get_behavior``.
        min_share: minimum share of finite samples within 100 units of the mode.

    Returns:
        The modal water position, or None when the Y or water channel is missing, no sample
        is finite, or the mode covers too small a share of the samples.
    """
    _, y_col = _xy_columns(bdata)
    if y_col is None or WATER_CHANNEL not in bdata:
        return None
    w = np.asarray(bdata[y_col], dtype=float) + np.asarray(
        bdata[WATER_CHANNEL], dtype=float
    )
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return None
    r = np.round(w / 100.0) * 100.0
    vals, counts = np.unique(r, return_counts=True)
    mode = float(vals[np.argmax(counts)])
    if np.mean(np.abs(r - mode) <= 100.0) < min_share:
        return None
    return mode


def extract_neural_and_position_hybrid(
    db: Database,
    session_name: str,
    session_config: dict,
    bin_width_sec: float = BIN_WIDTH_SEC,
    water_min_share: float = LINEAR_WATER_MIN_SHARE,
) -> tuple:
    """Extract binned spike counts, X/Y position and the water position for a whole session.

    Iterates over all trials for the session in chronological order.
    Neural data is binned at bin_width_sec resolution. Position is
    bin-averaged onto the same time axis as the neural bins. Every bin of a trial carries that
    trial's water position (``trial_water_position``), NaN when it is ill-defined or absent.

    A trial is skipped when its bin count is undefined (``_trial_bin_count``), or it has no
    spikes, no behavior document, or no X/Y channel. ``window_task_mask`` applies the same
    rules, so its bins stay aligned with these.

    Args:
        db:             pymongo Database.
        session_name:   e.g. ``"M21_250214"``.
        session_config: the session's entry from ``build_hybrid_neuron_mapping``; it fixes
                        the column of every neuron.
        bin_width_sec:  time bin width in seconds.
        water_min_share: ``trial_water_position``'s minimum share for a trial's water.

    Returns:
        (neural, x, y, water) where:
            neural: (n_bins_total, n_neurons) float32 array of spike rates
            x:      (n_bins_total,) float64 array of X positions
            y:      (n_bins_total,) float64 array of Y positions
            water:  (n_bins_total,) float64 water position of each bin's trial (NaN if none)
        Returns (None, None, None, None) if no usable trials are found.
    """
    date_id = session_name_to_date_id(db, session_name)
    trial_ids = [
        t["_id"]
        for t in db["trials"]
        .find({"date_db_id": date_id})
        .sort("trial_start_neural_data", 1)
    ]

    neuron_id_to_idx = session_config["neuron_id_to_idx"]
    n_neurons = session_config["n_total"]

    all_neural: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_water: list[np.ndarray] = []

    for trial_id in trial_ids:
        doc = db["trials"].find_one({"_id": trial_id})
        n_bins = _trial_bin_count(doc, bin_width_sec)
        if n_bins is None:
            continue

        ndata = get_neural(db, trial_id)
        if not ndata:
            continue

        trial_neural = np.zeros((n_bins, n_neurons), dtype=np.float32)
        for neuron_id, spikes in ndata.items():
            if neuron_id in neuron_id_to_idx:
                col_idx = neuron_id_to_idx[neuron_id]
                spikes_sec = np.array(spikes) / 10000  # 10 kHz sample index -> seconds
                counts, _ = np.histogram(
                    spikes_sec, bins=np.arange(n_bins + 1) * bin_width_sec
                )
                trial_neural[:, col_idx] = counts / bin_width_sec

        bdata = get_behavior(db, trial_id)
        if not bdata:
            continue

        x_col, y_col = _xy_columns(bdata)
        if x_col is None or y_col is None:
            continue

        x_data = np.array(bdata[x_col])
        y_data = np.array(bdata[y_col])

        # Behavior is sampled at its own (~90 Hz) rate: spread its samples evenly over
        # the trial's neural bins and average the samples that fall in each bin.
        behavior_bin_size = len(x_data) / n_bins
        trial_x = np.zeros(n_bins)
        trial_y = np.zeros(n_bins)

        for i in range(n_bins):
            start_idx = int(i * behavior_bin_size)
            end_idx = min(int((i + 1) * behavior_bin_size), len(x_data))
            if start_idx < end_idx:
                trial_x[i] = np.nanmean(x_data[start_idx:end_idx])
                trial_y[i] = np.nanmean(y_data[start_idx:end_idx])

        water = trial_water_position(bdata, water_min_share)
        all_neural.append(trial_neural)
        all_x.append(trial_x)
        all_y.append(trial_y)
        all_water.append(np.full(n_bins, np.nan if water is None else water))

    if not all_neural:
        return None, None, None, None

    return (
        np.vstack(all_neural),
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_water),
    )


def _classify_bin(x: float, y: float, paradigm: str, water: float) -> int:
    """Region label for a single position sample, dispatched on paradigm.

    Args:
        x:        lateral position.
        y:        position along the maze.
        paradigm: paradigm string; matched by substring like ``get_num_classes``.
        water:    the trial's water position, used by the Linear maze only.

    Returns:
        The region label, or -1 for a Linear-maze bin without a usable water position.
    """
    if "Y_Maze" in paradigm:
        return classify_point_ymaze(x, y)
    if "Nested" in paradigm:
        return classify_point_nested(x, y)
    return classify_point_more_water(y, water)


def create_windows_with_regions(
    neural_data: np.ndarray,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    window_size: int,
    paradigm: str,
    water: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a non-overlapping window over neural_data and assign region labels.

    Region label is taken from the position at the last bin of each window. Trailing bins
    that do not fill a whole window are dropped.

    Args:
        neural_data:  (n_bins, n_neurons) float32 spike rate array.
        x_positions:  (n_bins,) X coordinates.
        y_positions:  (n_bins,) Y coordinates.
        window_size:  bins per window (WINDOW_SIZE=10 → 1 s at 100 ms bins).
        paradigm:     maze type string.
        water:        (n_bins,) water position per bin (Linear maze only; NaN if none).

    Returns:
        ``(windows, regions)``: the (n_windows, window_size, n_neurons) windows and their
        (n_windows,) int region labels, with ``n_windows = n_bins // window_size``.
    """
    n_windows = len(neural_data) // window_size
    if water is None:
        water = np.full(len(neural_data), np.nan)
    windows = []
    regions = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        windows.append(neural_data[start:end])

        regions.append(
            _classify_bin(
                x_positions[end - 1], y_positions[end - 1], paradigm, water[end - 1]
            )
        )

    return np.array(windows), np.array(regions)


def window_task_mask(
    db: Database,
    session_name: str,
    window_size: int,
    bin_width_sec: float = BIN_WIDTH_SEC,
) -> np.ndarray:
    """Boolean mask (n_windows,) marking windows that lie entirely within a task trial.

    A task trial is one whose ``trial_status`` is ``"W"``. Trials are skipped by the same
    rules as ``extract_neural_and_position_hybrid``, so the mask lines up window for window
    with its output; the caller still checks the lengths match.

    Args:
        db:            pymongo Database.
        session_name:  e.g. ``"M50_250411"``.
        window_size:   bins per window (WINDOW_SIZE=10 -> 1 s at 100 ms bins).
        bin_width_sec: time bin width in seconds (must match the extraction).

    Returns:
        1-D boolean ndarray of length n_windows. True == task (W) window. Empty when no
        trial is usable.
    """
    date_id = session_name_to_date_id(db, session_name)
    trial_ids = [
        t["_id"]
        for t in db["trials"]
        .find({"date_db_id": date_id})
        .sort("trial_start_neural_data", 1)
    ]

    per_bin_status: list[np.ndarray] = []
    for trial_id in trial_ids:
        doc = db["trials"].find_one({"_id": trial_id})
        n_bins = _trial_bin_count(doc, bin_width_sec)
        if n_bins is None:
            continue

        if not get_neural(db, trial_id):
            continue
        bdata = get_behavior(db, trial_id)
        if not bdata:
            continue

        x_col, y_col = _xy_columns(bdata)
        if x_col is None or y_col is None:
            continue

        status = doc.get("trial_status")
        per_bin_status.append(np.full(n_bins, status == "W", dtype=bool))

    if not per_bin_status:
        return np.zeros(0, dtype=bool)

    bin_is_task = np.concatenate(per_bin_status)
    n_windows = len(bin_is_task) // window_size
    trimmed = bin_is_task[: n_windows * window_size].reshape(n_windows, window_size)
    return np.asarray(trimmed.all(axis=1), dtype=bool)


def window_region_labels(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    window_size: int,
    paradigm: str,
    water: np.ndarray | None = None,
    reduce: str = "last",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window region labels with a choice of temporal reduction over the bins.

    This helper classifies every bin in the window, then reduces the bin labels to one:

    - ``reduce="last"``: the last bin's label, identical to ``create_windows_with_regions``.
    - ``reduce="mode"``: the modal (most frequent) region across the window's bins; a tie
      goes to the smallest label.

    Args:
        x_positions: (n_bins,) X coordinates.
        y_positions: (n_bins,) Y coordinates.
        window_size: bins per window.
        paradigm:    maze type string.
        water:       (n_bins,) water position per bin (Linear maze only; NaN if none).
        reduce:      ``"last"`` or ``"mode"``; any other value behaves as ``"last"``.

    Returns:
        ``(regions, straddle)``, both of length ``n_windows = len(x_positions) //
        window_size``: the int64 labels, and a boolean flag for windows whose bins span
        more than one region.
    """
    n_windows = len(x_positions) // window_size
    if water is None:
        water = np.full(len(x_positions), np.nan)
    regions = np.empty(n_windows, dtype=np.int64)
    straddle = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        xs = x_positions[start:end]
        ys = y_positions[start:end]
        ws = water[start:end]
        bins_arr = np.asarray(
            [
                _classify_bin(x, y, paradigm, w)
                for x, y, w in zip(xs, ys, ws, strict=False)
            ]
        )
        straddle[i] = bins_arr.min() != bins_arr.max()
        if reduce == "mode":
            vals, counts = np.unique(bins_arr, return_counts=True)
            regions[i] = int(vals[np.argmax(counts)])
        else:  # "last": matches create_windows_with_regions exactly
            regions[i] = int(bins_arr[-1])

    return regions, straddle
