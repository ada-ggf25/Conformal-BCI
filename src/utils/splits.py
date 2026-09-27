"""Train/validation/test splitting utilities."""

from __future__ import annotations

import numpy as np


def chronological_task_split(
    task_mask: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological within-session split over task windows only.

    Non-task windows are left out entirely. The task windows are then partitioned into three
    contiguous, time-ordered blocks (defaults 80/10/10):

        train - Step 1 and Step 2 model training, plus fitting the learned scale sigma
        val   - early stopping / epoch selection and the conformal quantile q_hat
        test  - coverage evaluation

    Args:
        task_mask:  (n_windows,) boolean. True where the window is a task (W) window,
                    in chronological order.
        train_frac: fraction of task windows for training (default 0.80)
        val_frac:   fraction for validation / early stopping / q_hat (default 0.10)
        test_frac:  fraction for coverage evaluation (default 0.10)

    Returns:
        (train_idx, val_idx, test_idx) - three disjoint integer arrays indexing into the
        original window array, in chronological order.

    Raises:
        ValueError: if the three fractions do not sum to 1.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            "train/val/test fractions must sum to 1.0 "
            f"(got {train_frac + val_frac + test_frac})"
        )

    task_mask = np.asarray(task_mask, dtype=bool)
    task_idx = np.flatnonzero(task_mask)  # chronological positions of task windows
    n = len(task_idx)

    # Chronological block boundaries over the task windows.
    b1 = round(train_frac * n)
    b2 = round((train_frac + val_frac) * n)

    return task_idx[:b1], task_idx[b1:b2], task_idx[b2:]
