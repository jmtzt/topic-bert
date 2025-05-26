#!/bin/bash
export RUN_ID=$(.venv/bin/python src/predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
.venv/bin/python src/predict.py predict \
    --run-id $RUN_ID \
    --question-title "What is the capital of France?" \
    --question-content "I would like to know the capital city of France." \
    --best-answer "Paris is the capital of France."
