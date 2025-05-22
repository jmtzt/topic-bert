import datetime
import json
from collections import OrderedDict
from typing import Dict

import numpy as np
from ray.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function

from src import data, predict, utils
from src.config import NUM_TEST_SAMPLES, logger
from src.predict import TorchPredictor, get_best_run_id


def get_overall_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict:  # pragma: no cover
    """Get overall performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        Dict: overall metrics.
    """
    metrics = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    overall_metrics = {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }
    return overall_metrics


def get_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict
) -> Dict:  # pragma: no cover
    """Get per class performance metrics.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        class_to_index (Dict): dictionary mapping class to index.

    Returns:
        Dict: per class metrics.
    """
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    sorted_per_class_metrics = OrderedDict(
        sorted(
            per_class_metrics.items(),
            key=lambda topic: topic[1]["f1"],
            reverse=True,
        )
    )
    return sorted_per_class_metrics


@slicing_function()
def short_text(x):  # pragma: no cover
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 chars


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, ds: Dataset
) -> Dict:  # pragma: no cover
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.
        ds (Dataset): Ray dataset with labels.
    Returns:
        Dict: performance metrics for slices.
    """
    slice_metrics = {}
    df = ds.to_pandas()
    df["text"] = df.apply(data.combine_text_fields, axis=1)
    slices = PandasSFApplier([short_text]).apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            slice_metrics[slice_name] = {}
            slice_metrics[slice_name]["precision"] = metrics[0]
            slice_metrics[slice_name]["recall"] = metrics[1]
            slice_metrics[slice_name]["f1"] = metrics[2]
            slice_metrics[slice_name]["num_samples"] = len(y_true[mask])
    return slice_metrics


def evaluate(
    run_id: str = None,
    results_fp: str = None,
) -> Dict:  # pragma: no cover
    """Evaluate on the holdout dataset.

    Args:
        run_id (str): id of the specific run to load from. Defaults to None.
        results_fp (str, optional): location to save evaluation results to.
            Defaults to None.

    Returns:
        Dict: model's performance metrics on the dataset.
    """
    # Load test dataset from the huggingface hub
    ds, _ = data.load_data(split="test", num_samples=NUM_TEST_SAMPLES)
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # y_true
    preprocessor = predictor.get_preprocessor()
    preprocessed_ds = preprocessor.transform(ds)
    values = preprocessed_ds.select_columns(cols=["targets"]).take_all()
    y_true = np.stack([item["targets"] for item in values])

    # y_pred
    predictions = preprocessed_ds.map_batches(predictor).take_all()
    y_pred = np.array([d["output"] for d in predictions])

    # Metrics
    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_to_index=preprocessor.class_to_index,
        ),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, ds=ds),
    }
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":  # pragma: no cover
    # Example usage
    experiment_name = "bert_finetune_example"
    run_id = get_best_run_id(
        experiment_name=experiment_name,
        metric="val_loss",
        mode="ASC",
    )
    results_fp = f"results/eval_{experiment_name}_{run_id}.json"
    metrics = evaluate(
        run_id=run_id,
        results_fp=results_fp,
    )
