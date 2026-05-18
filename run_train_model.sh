#!/bin/bash

set -e

echo "Splitting data..."
python split.py data.csv --ratio 0.8 --seed 42

echo "Training model..."
python main.py

echo "Predicting on validation set..."
python predict.py --data validation.csv --model model.npz
