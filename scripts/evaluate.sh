#!/bin/bash
export EXPERIMENT_NAME="bert_finetune_example"
export RUN_ID=$(.venv/bin/python src/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
.venv/bin/python src/evaluate.py \
    --run-id $RUN_ID \
    --results-fp results/evaluation_results.json
