#!/bin/bash
export EXPERIMENT_NAME="bert_finetune_example"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
.venv/bin/python src/train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-workers 1 \
    --cpu-per-worker 8 \
    --gpu-per-worker 0 \
    --num-epochs 5 \
    --batch-size 64 \
    --results-fp results/training_results.json
