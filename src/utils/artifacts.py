"""Filesystem layout and write side of the pooled per-maze analysis bundles."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def paradigm_slug(paradigm: str) -> str:
    """Filesystem-safe directory name for a paradigm string.

    Runs of non-alphanumeric characters collapse to a single underscore, so
    ``'Nested_Maze+2'`` becomes ``'Nested_Maze_2'``.

    Raises:
        ValueError: if paradigm is empty or has no alphanumeric characters.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", paradigm).strip("_")
    if not slug:
        raise ValueError(
            f"paradigm {paradigm!r} yields an empty directory name; a concrete "
            "paradigm must be resolved before an artifact path is built"
        )
    return slug


def jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types.

    Numpy floats/ints/bools become Python scalars, arrays become nested lists, and
    non-finite floats (``inf`` / ``nan``) become ``None`` so ``json.dump`` succeeds.
    """
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def build_arrays_bundle(
    session_data: dict,
    valid_sessions: list[str],
    *,
    sigma: np.ndarray,
    u_raw: np.ndarray,
    signals: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Assemble the per-window ``arrays.npz``, all in ``valid_sessions`` order.

    Raises ``AssertionError`` if any field's row count disagrees (the contract).
    """
    order = valid_sessions
    split_code = {"train_idx": 0, "val_idx": 1, "test_idx": 2}  # -1 = dropped

    def _codes(s: str) -> np.ndarray:
        v = np.full(len(session_data[s]["regions"]), -1, dtype=int)
        for k, c in split_code.items():
            v[session_data[s][k]] = c
        return v

    arrays: dict[str, np.ndarray] = {
        "win_session_idx": np.concatenate(
            [np.full(len(session_data[s]["regions"]), i) for i, s in enumerate(order)]
        ),
        "win_region": np.concatenate(
            [session_data[s]["regions"] for s in order]
        ).astype(int),
        "win_split": np.concatenate([_codes(s) for s in order]),
        "win_probs": np.concatenate([session_data[s]["probs"] for s in order]).astype(
            np.float32
        ),
        "win_sigma": sigma.astype(np.float32),  # the learned conformal scale
    }
    for j, name in enumerate(signals):  # the four signals, named
        arrays[f"win_{name}"] = u_raw[:, j].astype(np.float32)

    n = len(arrays["win_region"])
    for k, v in arrays.items():
        assert len(v) == n, (
            f"arrays.npz row-order contract broken: {k} has {len(v)} rows, not {n}"
        )
    return arrays
