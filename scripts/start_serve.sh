#!/bin/bash
set -e

RUN_ID=$1
THRESHOLD=$2

if .venv/bin/ray status &>/dev/null; then
  .venv/bin/ray stop
fi

echo "Starting Ray cluster..."
.venv/bin/ray start --head

sleep 2

if ! .venv/bin/ray status &>/dev/null; then
  echo "ERROR: Ray failed to start properly"
  exit 1
fi

echo "Ray cluster started successfully"

echo "Starting serve application with run_id=${RUN_ID} and threshold=${THRESHOLD}..."
.venv/bin/python -m src.serve --run_id ${RUN_ID} --threshold ${THRESHOLD}

EXIT_CODE=$?

echo "Serve application exited with code $EXIT_CODE"

exit $EXIT_CODE
