# Multilayer Perceptron (MLP) from Scratch

A neural network implemented entirely from scratch using only NumPy (no TensorFlow/PyTorch). Built to classify breast tumors as **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer Dataset.

## Architecture

The codebase follows a clean layered design:

| File | Role |
|---|---|
| `multilayer_perceptron.py` | High-level API — configures and wraps the model |
| `model.py` | Training loop — epoch iteration, mini-batch SGD, early stopping |
| `network.py` | Core neural network — forward propagation & backpropagation |
| `optimizer.py` | SGD and Adam optimizers |
| `utils.py` | Math primitives — sigmoid, ReLU, softmax, cross-entropy, MSE |
| `main_train.py` | CLI entry point for training |
| `predict.py` | CLI entry point for inference |
| `split.py` | Dataset splitting utility |

## How It Works

1. **Data loading** (`main_train.py`): Reads `data.csv`, extracts 30 numeric features (columns 2+), one-hot encodes the label (`M` -> `[0,1]`, `B` -> `[1,0]`), and standardizes features (z-score normalization).

2. **Network construction** (`network.py`): Builds a fully-connected network with configurable hidden layers (default: 3 layers of 24 neurons). Weights are initialized with small random values (`* 0.1`).

3. **Forward pass**: Input flows through hidden layers with a configurable activation (ReLU or sigmoid), and the output layer uses **softmax** for 2-class probability output.

4. **Backward pass (backpropagation)**: Computes gradients via the chain rule. For cross-entropy + softmax, it uses the simplified gradient `y_pred - y_true`.

5. **Optimization** (`optimizer.py`): Supports **SGD** (vanilla gradient descent) and **Adam** (adaptive moment estimation with bias correction).

6. **Training** (`model.py`): Mini-batch training with shuffling, tracks loss/accuracy for both train and validation sets, and implements **early stopping** (restores best weights after patience runs out).

7. **Output**: Saves the trained model as `model.pkl` (via pickle) and generates a `learning_curve.png` plot showing loss and accuracy curves.

## Default Configuration

- Hidden layers: `[24, 24, 24]`
- Output: 2 neurons (binary classification via softmax)
- Loss: Cross-entropy
- Optimizer: Adam (lr=0.0001)
- Batch size: 4, Epochs: 200
- Early stopping patience: 10 epochs

## Installation

1. Set up the virtual environment:
   ```bash
   ./start_venv.sh
   source venv/bin/activate
   ```
   Or install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.1
- `pandas` >= 1.3.0

## Usage

### Training

```bash
python main_train.py --help
```

Example:
```bash
python main_train.py --data data.csv --layers 24 24 24 --epochs 200 --batch_size 4 --learning_rate 0.0001 --split 0.2 --solver adam
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to the training data CSV file | `data.csv` |
| `--layers` | Sizes of hidden layers | `24 24 24` |
| `--epochs` | Number of training epochs | `200` |
| `--batch_size` | Batch size for training | `4` |
| `--learning_rate` | Learning rate | `0.0001` |
| `--split` | Train/validation split ratio | `0.2` |
| `--solver` | Optimizer (`adam` or `sgd`) | `adam` |

The training script saves the trained model to `model.pkl` and generates a `learning_curve.png` plot.

### Prediction

```bash
python predict.py --data data.csv --model model.pkl
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to the input data CSV file | `data.csv` |
| `--model` | Path to the trained model file | `model.pkl` |
