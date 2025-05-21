import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from datasets import load_dataset
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from src import config
from src.config import logger


def load_data(
    dataset_name: str = config.DATASET_NAME,
    target_column: str = "topic",
    num_samples: int = None,
) -> Tuple[Dataset, List[str]]:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        num_samples (int, optional): The number of samples to load.
        Defaults to None.

    Returns:
        Tuple[Dataset, List[str]]: a Ray Dataset and list of class names.
    """
    ds = load_dataset(dataset_name, split="train")
    class_names = ds.features[target_column].names
    ds = ray.data.from_huggingface(ds)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds, class_names


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset to split.
        stratify (str): Name of column to split on.
        test_size (float): Proportion of dataset to split for test set.
        shuffle (bool, optional): whether to shuffle the dataset.
        Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 1234.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        """Naively split a dataframe into train and test splits.
        Add a column specifying whether it's the train or test split."""
        train, test = train_test_split(
            df, test_size=test_size, shuffle=shuffle, random_state=seed
        )
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(
        df: pd.DataFrame, split: str
    ) -> pd.DataFrame:  # pragma: no cover
        """Filter by data points that match the split column's value
        and return the dataframe with the _split column dropped."""
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = ds.groupby(stratify).map_groups(
        _add_split, batch_format="pandas"
    )  # group by each unique value in the column we want to stratify on
    train_ds = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "train"}, batch_format="pandas"
    )  # combine
    test_ds = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "test"}, batch_format="pandas"
    )  # combine

    # Shuffle each split (required)
    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def clean_text(text: str, stopwords: List = config.STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out.
        Defaults to STOPWORDS.

    Returns:
        str: cleaned text.
    """
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def tokenize(batch: Dict) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of data with the results of tokenization
        (`input_ids` and `attention_mask`) on the text inputs.
    """
    tokenizer = BertTokenizer.from_pretrained(
        config.PRETRAINED_MODEL_NAME, return_dict=False
    )
    max_len = tokenizer.model_max_length

    encoded_inputs = tokenizer(
        batch["text"].tolist(),
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    return dict(
        ids=encoded_inputs["input_ids"],
        masks=encoded_inputs["attention_mask"],
        targets=np.array(batch["topic"]),
    )


def combine_text_fields(row: Dict) -> str:
    """Combine text fields, handling empty values.

    Args:
        row (Dict): Dictionary containing the text fields

    Returns:
        str: Combined text with empty fields filtered out
    """
    texts = []
    if row.get("question_title"):
        texts.append(row["question_title"])
    if row.get("question_content"):
        texts.append(row["question_content"])
    if row.get("best_answer"):
        texts.append(row["best_answer"])
    return " ".join(texts)


def preprocess(df: pd.DataFrame) -> Dict:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """
    # feature engineering, combine text columns
    df["text"] = df.apply(combine_text_fields, axis=1)
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(
        columns=["id", "question_title", "question_content", "best_answer"],
        errors="ignore",
    )  # clean dataframe
    df = df[["text", "topic"]]  # rearrange columns
    outputs = tokenize(df)
    return outputs


class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or []
        self.class_to_index = {
            name: i for i, name in enumerate(self.class_names)
        }
        self.index_to_class = {
            i: name for i, name in enumerate(self.class_names)
        }

    def fit(self, ds):  # pragma: no cover
        # This is a no-op for this example, but you could use it to
        # e.g. convert class names to indices
        return self

    def transform(self, ds):
        return ds.map_batches(
            preprocess,
            batch_format="pandas",
        )


def plot_topic_distributions(
    train_counts: pd.DataFrame,
    val_counts: pd.DataFrame,
    class_names: List[str],
    save_dir: str = config.ROOT_DIR / "results",
):  # pragma: no cover
    """Create plots with the distribution of topics in train and val sets.

    Args:
        train_counts (pd.DataFrame): DataFrame with train set topic counts
        val_counts (pd.DataFrame): DataFrame with validation set topic counts
        class_names (List[str]): List of class names to use as labels
        save_dir (str, optional): Directory to save the plots.
        Defaults to ROOT_DIR/results.
    """

    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    color = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    train_counts["count()"].plot(kind="bar", ax=ax1, color=color)
    ax1.set_title("Topic Distribution in Training Set")
    ax1.set_xlabel("Topic")
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.legend().remove()

    val_counts["count()"].plot(kind="bar", ax=ax2, color=color)
    ax2.set_title("Topic Distribution in Validation Set")
    ax2.set_xlabel("Topic")
    ax2.set_ylabel("Count")
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "topic_distributions.png"))
    plt.close()


if __name__ == "__main__":  # pragma: no cover
    # Dataset
    num_samples = 10_000
    ds, class_names = load_data(num_samples=num_samples)
    train_ds, val_ds = stratify_split(ds, stratify="topic", test_size=0.2)
    tags = train_ds.unique(column="topic")
    num_classes = len(tags)

    # Preprocess
    preprocessor = CustomPreprocessor(class_names=class_names)
    logger.info(preprocessor.class_to_index)
    logger.info(preprocessor.index_to_class)
    # preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)

    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    logger.info(f"Train dataset: {train_ds.count()} samples")
    logger.info(f"Validation dataset: {val_ds.count()} samples")

    logger.info(f"Sample from train dataset: {train_ds.take(1)}")

    train_counts = train_ds.groupby("targets").count().to_pandas()
    val_counts = val_ds.groupby("targets").count().to_pandas()
    logger.info("Train dataset counts by topic:")
    logger.info(train_counts)
    logger.info("Validation dataset counts by topic:")
    logger.info(val_counts)

    plot_topic_distributions(train_counts, val_counts, class_names)
