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
	$(ACTIVATE) && isort src/ tests/

.PHONY: format-check
format-check:
	$(ACTIVATE) && ruff check src/ tests/
	$(ACTIVATE) && isort src/ tests/ --check-only

.PHONY: test
test:
	$(ACTIVATE) && PYTHONPATH=$(CURDIR) pytest tests/ --verbose --disable-warnings

.PHONY: clean
clean: format
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -rf .coverage*
