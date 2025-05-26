import argparse
from http import HTTPStatus
from typing import Dict
from fastapi.responses import JSONResponse
import ray

from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from src import predict
from src.config import MLFLOW_TRACKING_URI, mlflow
from src.predict import TorchPredictor
from src.evaluate import evaluate

app = FastAPI(
    title="BERT Topic Classifier",
    description="Classify topics of questions using a BERT model.",
    version="0.1",
)


@serve.deployment(
    num_replicas="1", ray_actor_options={"num_cpus": 4, "num_gpus": 0}
)
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        data = await request.json()
        results = evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @app.post("/predict/")
    async def _predict(self, request: Request):
        data = await request.json()
        sample_ds = ray.data.from_items(
            [
                {
                    "question_title": data.get("question_title", ""),
                    "question_content": data.get("question_content", ""),
                    "best_answer": data.get("best_answer", ""),
                    "topic": -1,
                }
            ]
        )
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]

            converted_probs = {}
            for key, value in prob.items():
                if hasattr(value, "item"):
                    converted_probs[key] = value.item()
                else:
                    converted_probs[key] = value

            results[i]["probabilities"] = converted_probs

            # Check threshold using converted value
            if converted_probs[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return JSONResponse(
            content={"results": results},
            media_type="application/json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="threshold for `other` class.",
    )
    args = parser.parse_args()
    ray.init()
    serve.run(
        ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold)
    )
