# Topic Classifier

A small project that fine-tunes a BERT model to classify topics of [Yahoo Questions](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) using Ray for distributed training and MLflow for experiment tracking.

## Set up

Install the project dependencies:

```bash
make install
```

To lint code:

```bash
make lint
```

To format code:

```bash
make format
```

Set up pre-commit hooks for automatic code quality checks:

```bash
pre-commit install
```

## Scripts

This project includes several scripts for different stages of the ML workflow:

### Training

Train the BERT model with specified hyperparameters:

```bash
./scripts/train.sh
```

This script runs distributed training using Ray with configurable parameters:
- Experiment name: `bert_finetune_example`
- Training configuration with dropout, learning rate, and scheduler settings
- Configurable number of workers, CPUs, and GPUs per worker
- Number of epochs and batch size

### Hyperparameter Tuning

Perform hyperparameter optimization to find the best model configuration:

```bash
./scripts/tune.sh
```

This script uses Ray Tune with HyperOpt search algorithm to optimize:
- Dropout probability (0.3 - 0.9)
- Learning rate (1e-5 - 5e-4)
- Learning rate scheduler factor and patience
- Runs multiple experiments with different configurations

### Experiment Tracking

Launch MLflow UI to track and compare experiments:

```bash
make mlflow
```

This starts the MLflow server at `http://localhost:8080` where you can:
- View experiment results and metrics
- Compare model performance across runs
- Download model artifacts and checkpoints
- Visualize training progress and hyperparameter effects

### Evaluation

Evaluate the best trained model on the test dataset:

```bash
./scripts/evaluate.sh
```

This script:
- Automatically finds the best model based on validation loss
- Runs evaluation on holdout test data
- Generates comprehensive metrics including per-class performance
- Saves results to `results/evaluation_results.json`

### Inference

Run inference on sample data using the best model:

```bash
./scripts/inference.sh
```

This script demonstrates single prediction on a sample question, showing:
- Topic classification prediction
- Confidence scores for each class
- Model loading and preprocessing pipeline

### Serving

Start a Ray Serve deployment for real-time predictions:

```bash
./scripts/start_serve.sh <RUN_ID> <THRESHOLD>
```

This launches a FastAPI server with endpoints:
- `/`: Health check
- `/predict/`: Topic classification for new questions
- `/run_id/`: Get the current model run ID

## Testing

Run the test suite:

```bash
make test
```

Run tests with coverage reporting:

```bash
make test-coverage
```

The test suite covers:
- Data loading and preprocessing functions
- Model architecture and prediction methods
- Utility functions and configuration
- Integration tests for the ML pipeline

## Prod

In a production environment, this setup would be extended with cloud-native infrastructure:

### Cloud Infrastructure

- **MLflow**: Deploy on cloud platforms (AWS SageMaker, Azure ML, GCP AI Platform) instead of local file storage
- **Artifact Storage**: Use S3, Azure Blob Storage, or GCS for model artifacts and datasets
- **Compute**: Submit training jobs to managed Ray clusters in Kubernetes

### Local Kubernetes Setup

For development and testing, you can emulate a production Kubernetes setup locally:

```bash
# Start minikube cluster
minikube start --cpus=8 --memory=16g

# Install KubeRay operator
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0

# Install ray kubectl plugin
kubectl krew update
kubectl krew install ray

# Create Ray cluster
kubectl ray create cluster ray-cluster-local --worker-replicas 2 --worker-cpu 3 --worker-memory 8Gi --head-cpu 1 --head-memory 3Gi

# Check cluster status
kubectl ray get cluster
kubectl get pods

# Connect to Ray cluster
kubectl ray session ray-cluster-local
```

Submit jobs via the Ray Job Submit SDK for distributed execution.

References:
- [Ray Kubernetes Guide](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/kubectl-plugin.html)
- [Krew](https://krew.sigs.k8s.io/docs/user-guide/setup/install/)

### CI/CD Pipeline

Currently implemented:
- Basic linting and formatting checks with Ruff
- Unit test execution with pytest
- Docker image building with multi-stage builds

Production enhancements would include:
- **Container Registry**: Push images to ECR/GCR/ACR for Kubernetes deployment
- **Automated Testing**: Integration tests, model validation, and performance benchmarks on a dedicated runner
- **Security Scanning**: Container vulnerability scanning and dependency audits
- **Deployment Automation**: GitOps workflows with ArgoCD or Flux

### Continual Learning

This foundation supports extending to continual learning workflows:

- **Scheduled Training**: Cron-based retraining pipelines
- **Data Pipelines**: Integration with feature stores and streaming data
- **Drift Detection**: Model monitoring and automatic retraining triggers
- **Online Evaluation**: A/B testing, canary rollouts, model comparisons
- **Production Comparison**: Direct experiment vs. production model comparison in PRs
