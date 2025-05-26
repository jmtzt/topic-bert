import json
import uuid

import pytest

from src import train
from src.config import NUM_TRAIN_SAMPLES, mlflow


def delete_experiment(experiment_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(
        experiment_name
    ).experiment_id
    client.delete_experiment(experiment_id=experiment_id)


@pytest.mark.training
def test_train_model():
    experiment_name = f"test_train-{uuid.uuid4().hex[:8]}"
    train_loop_config = {
        "dropout_p": 0.5,
        "lr": 1e-4,
        "lr_factor": 0.8,
        "lr_patience": 3,
    }
    result = train.train_model(
        experiment_name=experiment_name,
        train_loop_config=json.dumps(train_loop_config),
        num_workers=1,
        cpu_per_worker=8,
        gpu_per_worker=0,
        num_epochs=2,
        num_samples=NUM_TRAIN_SAMPLES,
        batch_size=64,
        results_fp=None,
    )
    delete_experiment(experiment_name=experiment_name)
    train_loss_list = result.metrics_dataframe.to_dict()["train_loss"]
    assert train_loss_list[0] > train_loss_list[1]  # loss decreased
