import tempfile
from typing import Tuple

import numpy as np
import ray.train as train
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


def eval_step(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
) -> Tuple[float, np.array, np.array]:
    """Eval step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between
        labels and predictions.

    Returns:
        Tuple[float, np.array, np.array]: cumulative loss,
        ground truths and preds.
    """
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=utils.collate_fn
    )
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(
                batch["targets"], num_classes=num_classes
            ).float()
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None:
    """Training loop that each ray worker will execute.

    Args:
        config (dict): arguments to use for training.
    """
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Get datasets
    utils.set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # Model
    base_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    model = FinetunedBert(
        base_model=base_model,
        dropout_p=dropout_p,
        embedding_dim=base_model.config.hidden_size,
        num_classes=num_classes,
    )
    model = train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    # Training
    num_workers = train.get_context().get_world_size()
    batch_size_per_worker = batch_size // num_workers
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(
            train_ds,
            batch_size_per_worker,
            model,
            num_classes,
            loss_fn,
            optimizer,
        )
        val_loss, _, _ = eval_step(
            val_ds, batch_size_per_worker, model, num_classes, loss_fn
        )
        scheduler.step(val_loss)

        # Checkpoint and report metrics
        with tempfile.TemporaryDirectory() as dp:
            # ddp, we can access the model under the module attribute
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model.module.save(dp=dp)
            else:
                model.save(dp=dp)
            metrics = dict(
                epoch=epoch,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                val_loss=val_loss,
            )
            checkpoint = train.Checkpoint.from_directory(dp)
            train.report(metrics, checkpoint=checkpoint)


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
    batch_size = 2

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
        batch_size,
        model,
        num_classes,
        loss_fn,
        optimizer,
    )
    logger.info(f"Train loss: {train_loss:.4f}")

    val_loss, y_trues, y_preds = eval_step(
        val_ds, batch_size, model, num_classes, loss_fn
    )

    logger.info(f"Validation loss: {val_loss:.4f}")
