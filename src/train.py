import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.data import Dataset
from transformers import BertModel

from src import data, utils
from src.config import PRETRAINED_MODEL_NAME, logger
from src.models import FinetunedBert


def train_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between
        labels and predictions.
        optimizer (torch.optimizer.Optimizer): optimizer to use for updating
        the model's weights.

    Returns:
        float: cumulative loss for the dataset.
    """
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=utils.collate_fn
    )
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets = F.one_hot(
            batch["targets"], num_classes=num_classes
        ).float()  # one-hot (for loss_fn)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss


if __name__ == "__main__":
    # Dataset
    num_samples = 100
    ds, class_names = data.load_data(num_samples=num_samples)
    train_ds, val_ds = data.stratify_split(ds, stratify="topic", test_size=0.2)

    preprocessor = data.CustomPreprocessor()
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Model
    base_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    dropout_p = 0.1
    num_classes = len(class_names)

    lr = 1e-4

    model = FinetunedBert(
        base_model=base_model,
        dropout_p=dropout_p,
        embedding_dim=base_model.config.hidden_size,
        num_classes=num_classes,
    )
    # model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = train_step(
        train_ds,
        2,
        model,
        num_classes,
        loss_fn,
        optimizer,
    )
    logger.info(f"Train loss: {train_loss:.4f}")
