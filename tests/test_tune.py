import json
import uuid

from config import NUM_TRAIN_SAMPLES
import pytest
from test_train import delete_experiment

from src import tune


@pytest.mark.training
def test_tune_models():
    num_runs = 2
    experiment_name = f"test_tune-{uuid.uuid4().hex[:8]}"
    initial_params = [
        {
            "train_loop_config": {
                "dropout_p": 0.5,
                "lr": 1e-4,
                "lr_factor": 0.8,
                "lr_patience": 3,
            }
        }
    ]
    results = tune.tune_models(
        experiment_name=experiment_name,
        initial_params=json.dumps(initial_params),
        num_workers=1,
        cpu_per_worker=8,
        gpu_per_worker=0,
        num_runs=num_runs,
        num_epochs=1,
        num_samples=NUM_TRAIN_SAMPLES,
        batch_size=64,
        results_fp=None,
    )
    delete_experiment(experiment_name=experiment_name)
    assert len(results.get_dataframe()) == num_runs
