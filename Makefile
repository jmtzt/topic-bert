# Makefile
SHELL = /bin/bash
VENV_DIR = .venv
PYTHON = python3
ACTIVATE = source $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR) && \
	$(ACTIVATE) && \
	pip install uv

install: $(VENV_DIR)/bin/activate
	$(ACTIVATE) && uv sync --all-extras

lint:
	$(ACTIVATE) && ruff check src/ tests/ --fix

format:
	$(ACTIVATE) && isort src/ tests/
	$(ACTIVATE) && ruff format src/ tests/

format-check:
	$(ACTIVATE) && isort src/ tests/ --check-only
	$(ACTIVATE) && ruff check src/ tests/

test:
	$(ACTIVATE) && pytest tests/ --verbose --disable-warnings

clean: format
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	rm -rf .coverage*
