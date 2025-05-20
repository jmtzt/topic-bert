import json
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from src.config import PRETRAINED_MODEL_NAME, logger


class FinetunedBert(nn.Module):
    def __init__(self, base_model, dropout_p, embedding_dim, num_classes):
        super(FinetunedBert, self).__init__()
        self.base_model = base_model
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        out = self.base_model(input_ids=ids, attention_mask=masks)
        z = self.dropout(out.pooler_output)
        z = self.fc1(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, batch):
        self.eval()
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            contents = {
                "dropout_p": self.dropout_p,
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
        base_model = BertModel.from_pretrained(
            PRETRAINED_MODEL_NAME, return_dict=False
        )
        model = cls(base_model=base_model, **kwargs)
        model.load_state_dict(
            torch.load(state_dict_fp, map_location=torch.device("cpu"))
        )
        return model


def create_random_batch(
    batch_size: int = 3, max_seq_len: int = 10
) -> Dict[str, torch.Tensor]:
    """Create a random batch of tokenized inputs for testing.

    Args:
        batch_size (int): Number of sequences in the batch
        max_seq_len (int): Maximum sequence length

    Returns:
        Dict[str, torch.Tensor]: Batch of tokenized inputs
    """
    ids = torch.randint(0, 1000, (batch_size, max_seq_len))
    masks = torch.ones_like(ids)
    targets = torch.randint(0, 10, (batch_size,))

    return {"ids": ids, "masks": masks, "targets": targets}


if __name__ == "__main__":
    # Example usage
    base_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)

    model = FinetunedBert(
        base_model=base_model,
        dropout_p=0.1,
        embedding_dim=768,
        num_classes=10,
    )
    logger.info("Model initialized successfully.")
    logger.info(model)

    batch = create_random_batch()

    outputs = model(batch)
    logger.info(f"Output shape: {outputs.shape}")

    predictions = model.predict(batch)
    logger.info(f"Predictions: {predictions}")

    probabilities = model.predict_proba(batch)
    logger.info(f"Probabilities shape: {probabilities.shape}")
    logger.info(f"Sample probabilities: {probabilities[0]}")
