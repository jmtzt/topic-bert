#!/bin/bash
export EXPERIMENT_NAME="bert_hparam_tune_example"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
.venv/bin/python src/tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 2 \
    --num-workers 1 \
    --cpu-per-worker 8 \
    --gpu-per-worker 1 \
    --num-epochs 5 \
    --batch-size 64 \
    --results-fp results/tuning_results.json
