import numpy as np
import torch
from ray.train.torch import get_device

from src import utils


def test_pad_array():
    arr = np.array([[1, 2], [1, 2, 3]], dtype="object")
    padded_arr = np.array([[1, 2, 0], [1, 2, 3]])
    assert np.array_equal(utils.pad_array(arr), padded_arr)


def test_collate_fn():
    batch = {
        "ids": np.array([[1, 2], [1, 2, 3]], dtype="object"),
        "masks": np.array([[1, 1], [1, 1, 1]], dtype="object"),
        "targets": np.array([3, 1]),
    }
    processed_batch = utils.collate_fn(batch)
    expected_batch = {
        "ids": torch.as_tensor(
            [[1, 2, 0], [1, 2, 3]], dtype=torch.int32, device=get_device()
        ),
        "masks": torch.as_tensor(
            [[1, 1, 0], [1, 1, 1]], dtype=torch.int32, device=get_device()
        ),
        "targets": torch.as_tensor(
            [3, 1], dtype=torch.int64, device=get_device()
        ),
    }
    for k in batch:
        assert torch.allclose(processed_batch[k], expected_batch[k])
