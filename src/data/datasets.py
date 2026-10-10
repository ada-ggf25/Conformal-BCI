"""PyTorch datasets and the collate function that feed the two training steps.

``HybridClassificationDataset`` and ``hybrid_collate_fn`` serve Step 1 (the BC-VAE encoder),
which trains on single windows grouped by session. ``LatentSequenceDataset`` serves Step 2
(the alignment decoder), which trains on short sequences of pre-extracted Step 1 latents.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class HybridClassificationDataset(Dataset):
    """Flat dataset of (window, region, session) samples for Step 1 BC-VAE training.

    Every selected window of every session becomes one sample, so a shuffled loader mixes
    sessions within a batch; ``hybrid_collate_fn`` then regroups each batch by session.

    Args:
        session_data_dict:  ``{session_name: data}``, where ``data`` holds ``"windows"``
                            (n_windows, T, n_neurons), ``"regions"`` (n_windows,), the split
                            index arrays, and optionally ``"windows_raw"`` (the spike counts
                            before normalization, same shape as ``"windows"``).
        index_key:          which split index to use (``"train_idx"`` or ``"val_idx"``).
    """

    def __init__(
        self,
        session_data_dict: dict,
        index_key: str = "train_idx",
    ):
        self.samples: list = []

        for session_name, data in session_data_dict.items():
            windows = data["windows"][data[index_key]]
            regions = data["regions"][data[index_key]]
            # Raw spike counts (pre-normalization) for the Poisson recon target.
            # Falls back to the (possibly z-scored) windows when not provided.
            windows_raw = data.get("windows_raw", data["windows"])[data[index_key]]
            for i in range(len(windows)):
                self.samples.append(
                    {
                        "window": windows[i],
                        "window_raw": windows_raw[i],
                        "region": regions[i],
                        "session_name": session_name,
                    }
                )

    def __len__(self) -> int:
        """Number of samples: the selected windows summed over every session."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """One sample as float32 tensors plus its region label and session name.

        Returns:
            ``{"window", "window_raw", "region", "session_name"}``; both windows are
            (T, n_neurons) float32 tensors.
        """
        sample = self.samples[idx]
        return {
            "window": torch.tensor(sample["window"], dtype=torch.float32),
            "window_raw": torch.tensor(sample["window_raw"], dtype=torch.float32),
            "region": sample["region"],
            "session_name": sample["session_name"],
        }


def hybrid_collate_fn(batch: list[dict]) -> dict:
    """Collate a batch from HybridClassificationDataset into per-session dicts.

    Grouping by session name is what carries session identity through Step 1, so no
    session index is emitted alongside it. Each session needs its own group because the
    encoder's input projection and read-out are per session.

    Args:
        batch: samples as returned by ``HybridClassificationDataset.__getitem__``.

    Returns:
        ``{session_name: {"windows", "windows_raw", "regions"}}``: the two stacked
        (n, T, n_neurons) float tensors and an (n,) long tensor of region labels, with
        ``n`` the session's share of the batch.
    """
    session_groups: dict = {}
    for sample in batch:
        name = sample["session_name"]
        if name not in session_groups:
            session_groups[name] = {
                "windows": [],
                "windows_raw": [],
                "regions": [],
            }
        session_groups[name]["windows"].append(sample["window"])
        session_groups[name]["windows_raw"].append(sample["window_raw"])
        session_groups[name]["regions"].append(sample["region"])

    collated = {}
    for name, group in session_groups.items():
        collated[name] = {
            "windows": torch.stack(group["windows"]),
            "windows_raw": torch.stack(group["windows_raw"]),
            "regions": torch.tensor(group["regions"], dtype=torch.long),
        }
    return collated


class LatentSequenceDataset(Dataset):
    """Dataset of front-padded latent sequences for Step 2 alignment training.

    Builds sequences from pre-extracted Step 1 latents. The sequence for window ``i`` is the
    ``seq_len`` latents ending at ``i``. The session's first latent is repeated in front,
    so early windows still get a full-length sequence (the same padding as
    ``build_latent_sequences``). Context is taken from the whole session, so a selected
    window's sequence may include latents of windows outside ``index_key``.

    Args:
        session_data:       ``{session_name: data}``, where ``data`` holds
                            ``"step1_all_latents"`` (n_windows, latent_dim) for every window
                            of the session, ``"regions"`` (n_windows,), and the split index
                            arrays.
        valid_sessions:     ordered list of session names to include. A session's position
                            in this list is the session id the dataset emits.
        seq_len:            sliding window length. Required: pass ``STEP2_SEQ_LEN``
                            rather than duplicating its value here, since importing it
                            would make ``src/data/`` depend on ``src/models/``.
        index_key:          which split to use (``"train_idx"`` or ``"val_idx"``).
    """

    def __init__(
        self,
        session_data: dict,
        valid_sessions: list[str],
        seq_len: int,
        index_key: str = "train_idx",
    ):
        sequences_list: list = []
        regions_list: list = []
        session_ids_list: list = []
        session_name_to_id = {name: i for i, name in enumerate(valid_sessions)}

        for session_name in valid_sessions:
            data = session_data[session_name]
            all_latents = data["step1_all_latents"]
            regions = data["regions"]
            keep_idx = set(data[index_key].tolist())
            sid = session_name_to_id[session_name]

            # Front-pad with seq_len - 1 copies of the first latent so that window i's
            # sequence is padded[i : i + seq_len], ending at latent i.
            padded = np.concatenate(
                [np.tile(all_latents[0:1], (seq_len - 1, 1)), all_latents], axis=0
            )

            for i in range(len(all_latents)):
                if i not in keep_idx:
                    continue
                sequences_list.append(padded[i : i + seq_len])
                regions_list.append(regions[i])
                session_ids_list.append(sid)

        self.sequences = np.array(sequences_list, dtype=np.float32)
        self.regions = np.array(regions_list, dtype=np.int64)
        self.session_ids = np.array(session_ids_list, dtype=np.int64)

    def __len__(self) -> int:
        """Number of sequences: the selected windows summed over every session."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """One ``(sequence, region, session_id)`` triple.

        Returns:
            A (seq_len, latent_dim) float32 tensor, the window's region label, and the
            session's index in ``valid_sessions``.
        """
        return (
            torch.from_numpy(self.sequences[idx]),
            self.regions[idx],
            self.session_ids[idx],
        )
