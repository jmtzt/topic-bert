from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import ray

from src import data


@pytest.fixture(scope="module")
def df():
    data = [
        {
            "question_title": "a0",
            "question_content": "b0",
            "best_answer": "c0",
            "topic": "d0",
        }
    ]
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def class_to_index():
    class_to_index = {"d0": 0, "t1": 1}
    return class_to_index


@pytest.fixture(scope="module")
def mock_dataset():
    mock_data = [
        {
            "question_title": "title1",
            "question_content": "content1",
            "best_answer": "answer1",
            "topic": "t1",
        },
        {
            "question_title": "title2",
            "question_content": "content2",
            "best_answer": "answer2",
            "topic": "t2",
        },
    ]
    return ray.data.from_items(mock_data)


@pytest.fixture(scope="module")
def mock_class_names():
    return ["t1", "t2"]


def test_load_data(mock_dataset, mock_class_names):
    with patch("src.data.load_data") as mock_load:
        mock_load.return_value = (mock_dataset, mock_class_names)
        ds, class_names = data.load_data(num_samples=2)
        assert ds.count() == 2
        assert isinstance(class_names, list)
        assert len(class_names) == 2
        assert all(isinstance(name, str) for name in class_names)
        assert set(class_names) == {"t1", "t2"}


def test_stratify_split():
    n_per_class = 10
    topics = n_per_class * ["t1"] + n_per_class * ["t2"]
    ds = ray.data.from_items([dict(topic=t) for t in topics])
    train_ds, test_ds = data.stratify_split(
        ds, stratify="topic", test_size=0.5
    )
    train_target_counts = train_ds.to_pandas().topic.value_counts().to_dict()
    test_target_counts = test_ds.to_pandas().topic.value_counts().to_dict()
    assert train_target_counts == test_target_counts


@pytest.mark.parametrize(
    "text, sw, clean_text",
    [
        ("hi", [], "hi"),
        ("hi you", ["you"], "hi"),
        ("hi yous", ["you"], "hi yous"),
    ],
)
def test_clean_text(text, sw, clean_text):
    assert data.clean_text(text=text, stopwords=sw) == clean_text


def test_preprocess(df):
    assert "text" not in df.columns
    outputs = data.preprocess(df)
    assert set(outputs) == {"ids", "masks", "targets"}


def test_transform(mock_dataset, mock_class_names):
    with patch("src.data.load_data") as mock_load:
        mock_load.return_value = (mock_dataset, mock_class_names)
        ds, class_names = data.load_data()
        preprocessor = data.CustomPreprocessor(class_names=class_names)
        preprocessed_ds = preprocessor.transform(ds)
        assert len(preprocessor.class_to_index) == 2
        assert ds.count() == preprocessed_ds.count()


@pytest.mark.parametrize(
    "row, expected",
    [
        (
            {
                "question_title": "Title",
                "question_content": "Content",
                "best_answer": "Answer",
            },
            "Title Content Answer",
        ),
        (
            {
                "question_title": "Title",
                "question_content": "",
                "best_answer": "Answer",
            },
            "Title Answer",
        ),
        (
            {
                "question_title": "",
                "question_content": "Content",
                "best_answer": "",
            },
            "Content",
        ),
        (
            {"question_title": "", "question_content": "", "best_answer": ""},
            "",
        ),
        ({}, ""),
    ],
)
def test_combine_text_fields(row, expected):
    result = data.combine_text_fields(row)
    assert result == expected


@patch("src.data.BertTokenizer.from_pretrained")
def test_tokenize(mock_tokenizer_class):
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 512
    mock_tokenizer.return_value = {
        "input_ids": np.array([[101, 102, 103]]),
        "attention_mask": np.array([[1, 1, 1]]),
    }
    mock_tokenizer_class.return_value = mock_tokenizer

    batch = {"text": pd.Series(["test text"]), "topic": [0]}

    result = data.tokenize(batch)

    assert "ids" in result
    assert "masks" in result
    assert "targets" in result

    mock_tokenizer.assert_called_once_with(
        ["test text"],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
