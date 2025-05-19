import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from datasets import load_dataset
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src.config import DATASET_NAME, PRETRAINED_MODEL_NAME, STOPWORDS, logger


def load_data(
    dataset_name: str = DATASET_NAME,
    target_column: str = "topic",
    num_samples: int = None,
) -> Tuple[Dataset, List[str]]:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        Tuple[Dataset, List[str]]: Our dataset represented by a Ray Dataset and list of class names.
    """
    ds = load_dataset(dataset_name, split="train")
    class_names = ds.features[target_column].names
    ds = ray.data.from_huggingface(ds)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds, class_names
