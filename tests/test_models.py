import os
import tempfile

import pytest
import torch
from transformers import BertModel

from src.config import PRETRAINED_MODEL_NAME
from src.models import FinetunedBert, create_random_batch


@pytest.fixture
def base_model():
    return BertModel.from_pretrained(PRETRAINED_MODEL_NAME)


@pytest.fixture
def model(base_model):
    return FinetunedBert(
        base_model=base_model, dropout_p=0.1, embedding_dim=768, num_classes=10
    )


@pytest.fixture
def batch():
    return create_random_batch(batch_size=2, max_seq_len=5)


def test_model_initialization(model):
    """Test if model initializes with correct attributes."""
    assert model.dropout_p == 0.1
    assert model.embedding_dim == 768
    assert model.num_classes == 10
    assert isinstance(model.dropout, torch.nn.Dropout)
    assert isinstance(model.fc1, torch.nn.Linear)
    assert model.fc1.out_features == 10


def test_forward_pass(model, batch):
    """Test if forward pass returns correct shape."""
    outputs = model(batch)
    assert outputs.shape == (2, 10)  # batch_size x num_classes


def test_predict(model, batch):
    """Test if predict method returns correct shape and values."""
    predictions = model.predict(batch)
    assert predictions.shape == (2,)  # batch_size
    # predictions within range
    assert all(0 <= pred < 10 for pred in predictions)


def test_predict_proba(model, batch):
    """Test if predict_proba returns correct shape and valid probabilities."""
    probabilities = model.predict_proba(batch)
    assert probabilities.shape == (2, 10)  # batch_size x num_classes
    # probs sum to 1
    assert torch.allclose(
        torch.tensor(probabilities.sum(axis=1)), torch.ones(2)
    )
    assert torch.all(torch.tensor(probabilities) >= 0)  # all probs >= 0


@torch.inference_mode()
def test_save_and_load(model, batch):
    """Test if model can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model.eval()
        model.save(temp_dir)

        assert os.path.exists(os.path.join(temp_dir, "args.json"))
        assert os.path.exists(os.path.join(temp_dir, "model.pt"))

        loaded_model = FinetunedBert.load(
            args_fp=os.path.join(temp_dir, "args.json"),
            state_dict_fp=os.path.join(temp_dir, "model.pt"),
        )

        loaded_model.eval()

        original_output = model(batch)
        loaded_output = loaded_model(batch)
        assert torch.allclose(original_output, loaded_output)


def test_create_random_batch():
    """Test if create_random_batch returns correct shapes and types."""
    batch = create_random_batch(batch_size=3, max_seq_len=8)

    assert isinstance(batch, dict)
    assert "ids" in batch
    assert "masks" in batch
    assert "targets" in batch

    assert batch["ids"].shape == (3, 8)
    assert batch["masks"].shape == (3, 8)
    assert batch["targets"].shape == (3,)

    assert torch.all(batch["masks"] == 1)  # all masks should be 1
    assert torch.all(batch["ids"] >= 0)  # all ids should be non-negative
    assert torch.all(
        batch["targets"] >= 0
    )  # all targets should be non-negative
