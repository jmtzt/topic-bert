# Makefile
SHELL = /bin/bash
VENV_DIR = .venv
PYTHON = python3
ACTIVATE = source $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR) && \
	$(ACTIVATE) && \
	pip install uv

.PHONY: install
install: $(VENV_DIR)/bin/activate
	$(ACTIVATE) && uv sync --all-extras && uv pip install -e .

.PHONY: lint
lint:
	$(ACTIVATE) && ruff check src/ tests/ --fix

.PHONY: format
format:
	$(ACTIVATE) && ruff format src/ tests/

.PHONY: format-check
format-check:
	$(ACTIVATE) && ruff check src/ tests/

.PHONY: test
test:
	$(ACTIVATE) && PYTHONPATH=$(CURDIR) pytest tests/ --verbose --disable-warnings

.PHONY: test-coverage
test-coverage:
	$(ACTIVATE) && PYTHONPATH=$(CURDIR) pytest tests/ --verbose --disable-warnings --cov=src --cov-report=term --cov-report=term-missing

.PHONY: mlflow
mlflow:
	$(ACTIVATE) && \
	export MODEL_REGISTRY=$$(python -c "from src import config; print(config.MLFLOW_TRACKING_URI.replace('file://', ''))") && \
	mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $$MODEL_REGISTRY

.PHONY: clean
clean: format
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -rf .coverage*
