import datetime
import json

import ray
from ray import tune
from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import DataConfig
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

from src import data, train, utils
from src.config import EFS_DIR, MLFLOW_TRACKING_URI, logger


def tune_models(
    experiment_name: str,
    initial_params: str,
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_runs: int = 1,
    num_samples: int = None,
    num_epochs: int = None,
    batch_size: int = None,
    results_fp: str = None,
) -> ray.tune.result_grid.ResultGrid:
    """Hyperparameter tuning experiment.

    Args:
        experiment_name (str): name of the experiment for this training
            workload.
        dataset_loc (str): location of the dataset.
        initial_params (str): initial config for the tuning workload.
        num_workers (int, optional): number of workers to use for training.
            Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker.
            Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker.
            Defaults to 0.
        num_runs (int, optional): number of runs in this tuning experiment.
            Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config.
            Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config.
            Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config.
            Defaults to None.
        results_fp (str, optional): filepath to save the tuning results.
            Defaults to None.

    Returns:
        ray.tune.result_grid.ResultGrid: results of the tuning experiment.
    """
    # Set up
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # Dataset
    ds, class_names = data.load_data(
        num_samples=train_loop_config.get("num_samples", None),
    )
    train_ds, val_ds = data.stratify_split(ds, stratify="topic", test_size=0.2)
    train_loop_config["num_classes"] = len(class_names)

    # Dataset config
    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(
        datasets_to_split=["train"], execution_options=options
    )

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)

    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        metadata={"class_to_index": preprocessor.class_to_index},
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # Run configuration
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
        storage_path=EFS_DIR,
    )

    # Hyperparameters to start with
    initial_params = json.loads(initial_params)
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=2
    )  # trade off b/w optimization and search space

    # Parameter space
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # Scheduler
    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],
        grace_period=1,  # min epoch per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )

    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(
            experiment_name=experiment_name,
            trial_id=best_trial.metrics["trial_id"],
        ),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(
            best_trial.metrics_dataframe.to_dict(),
            keys=["epoch", "train_loss", "val_loss"],
        ),
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:  # pragma: no cover
        utils.save_dict(d, results_fp)
    return results


if __name__ == "__main__":  # pragma: no cover
    # Example usage
    experiment_name = "bert_hparam_tune_test"
    train_loop_config = {
        "dropout_p": 0.5,
        "lr": 1e-4,
        "lr_factor": 0.8,
        "lr_patience": 3,
    }

    initial_params = [{"train_loop_config": train_loop_config}]

    tune_models(
        experiment_name=experiment_name,
        initial_params=json.dumps(initial_params),
        num_workers=1,
        cpu_per_worker=8,
        gpu_per_worker=0,
        num_samples=100,
        batch_size=64,
        num_runs=1,
        num_epochs=5,
    )
