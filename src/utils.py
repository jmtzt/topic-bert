from typing import Dict

import numpy as np
import torch
from ray.train.torch import get_device


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Pad an 2D array with zeros until all rows in the
    2D array are of the same length as a the longest
    row in the 2D array.

    Args:
        arr (np.array): input array

    Returns:
        np.array: zero padded array
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(
    batch: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """Convert a batch of numpy arrays to tensors (with appropriate padding).

    Args:
        batch (Dict[str, np.ndarray]): input batch as a dictionary of numpy arrays.

    Returns:
        Dict[str, torch.Tensor]: output batch as a dictionary of tensors.
    """
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(
            array, dtype=dtypes[key], device=get_device()
        )
    return tensor_batch
