"""PyTorch Dataset classes and collation utilities for the pipeline.

Adapted from the Liu Lab's cross-session alignment codebase.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class HybridClassificationDataset(Dataset):
    """Flat dataset of (window, region, session) samples for Step 1 BC-VAE training.

    Args:
        session_data_dict:  dict[session_name → {'windows': ndarray, 'regions': ndarray,
                            'train_idx': ndarray, ...}]
        index_key:          which split index to use ('train_idx' or 'val_idx')
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
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
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
    session index is emitted alongside it.

    Returns:
        dict[session_name → {'windows': Tensor, 'windows_raw': Tensor,
                             'regions': LongTensor}]
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

    Builds sequences from pre-extracted Step 1 latents.

    Args:
        session_data:       dict[session_name → {'step1_all_latents': ndarray,
                            'regions': ndarray, 'train_idx': ndarray, ...}]
        valid_sessions:     ordered list of session names to include
        seq_len:            sliding window length. Required: pass ``STEP2_SEQ_LEN``
                            rather than duplicating its value here, since importing it
                            would make ``src/data/`` depend on ``src/models/``.
        index_key:          which split to use ('train_idx' or 'val_idx')
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
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        return (
            torch.from_numpy(self.sequences[idx]),
            self.regions[idx],
            self.session_ids[idx],
        )
