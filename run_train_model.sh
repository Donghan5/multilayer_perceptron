#!/bin/bash

set -e

SOLVER="${1:-adam}"

if [[ "$SOLVER" != "adam" && "$SOLVER" != "sgd" ]]; then
    echo "Error: solver must be 'adam' or 'sgd' (got '$SOLVER')"
    echo "Usage: $0 [adam|sgd]"
    exit 1
fi

echo "Splitting data..."
python split.py data.csv --ratio 0.8 --seed 42

echo "Training model (solver=$SOLVER)..."
if [[ "$SOLVER" == "sgd" ]]; then
    python main.py --solver "$SOLVER" --learning_rate 0.001
else
    python main.py --solver "$SOLVER"
fi

echo "Predicting on validation set..."
python predict.py --data validation.csv --model model.npz
