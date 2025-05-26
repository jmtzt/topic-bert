FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ ./src/
COPY Makefile .

RUN mkdir -p logs efs results

RUN make install

# FastAPI
EXPOSE 8000

# Mlflow
EXPOSE 8080

ENV PYTHONPATH=/app

CMD [".venv/bin/python", "-m", "src.serve", "--run_id", "${RUN_ID}", "--threshold", "0.9"]
