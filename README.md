# Multilayer Perceptron (MLP) from Scratch

A neural network implemented entirely from scratch using only NumPy (no TensorFlow/PyTorch). Built to classify breast tumors as **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer Dataset.

## Architecture

The codebase follows a clean layered design:

| File | Role |
|---|---|
| `model.py` | Training loop — epoch iteration, mini-batch training, early stopping, standardization, serialization |
| `network.py` | Core neural network — forward propagation & backpropagation |
| `optimizer.py` | SGD and Adam optimizers |
| `utils.py` | Math primitives — sigmoid, ReLU, softmax, cross-entropy, one-hot encoding |
| `main.py` | CLI entry point for training |
| `predict.py` | CLI entry point for inference |

## How It Works

1. **Data loading** (`main.py`): Reads pre-split CSV files (`train.csv`, `validation.csv`), extracts 30 numeric features (columns 2+), and one-hot encodes the label (`M` -> `[0,1]`, `B` -> `[1,0]`).

2. **Standardization** (`model.py`): Z-score normalization is applied inside `Model.fit()`. Training mean/std are stored in the model and reused at inference time to prevent data leakage.

3. **Network construction** (`network.py`): Builds a fully-connected network with configurable hidden layers (default: 3 layers of 24 neurons). Weights are initialized with **He Uniform** (default) or **Xavier Uniform** via `--weights_initializer`.

4. **Forward pass**: Input flows through hidden layers with a configurable activation (ReLU or sigmoid), and the output layer uses **softmax** for 2-class probability output.

5. **Backward pass (backpropagation)**: Computes gradients via the chain rule. For cross-entropy + softmax, it uses the simplified gradient `(y_pred - y_true) / batch_size`.

6. **Optimization** (`optimizer.py`): Supports **SGD** (vanilla gradient descent) and **Adam** (adaptive moment estimation with bias correction).

7. **Training** (`model.py`): Mini-batch training with shuffling, tracks loss/accuracy for both train and validation sets, and implements **early stopping** (restores best weights after patience runs out).

8. **Output**: Saves the trained model as `model.pkl` (via pickle) and generates a `learning_curve.png` plot showing loss and accuracy curves.

## Design Decisions

**Why softmax + cross-entropy only?**
The combined derivative simplifies to `(y_pred - y_true)`, which is numerically stable and cheap to compute. Softmax + MSE introduces the full softmax Jacobian into the gradient, risking vanishing gradients. Unsupported combinations raise an error to prevent silent failures.

**Why raise on unsupported activation/loss combinations?**
The mandatory requirement is softmax + cross-entropy only. Rather than maintaining untested code paths and risking subtle bugs, the implementation refuses unsupported combinations explicitly. Fail-fast over false flexibility.

**Why ReLU as the default hidden activation?**
ReLU is the modern default for deep networks. On small datasets sigmoid can be competitive (measured: sigmoid val_acc 97.4% vs ReLU 95%). Both are supported via the `--activation` flag.

**Why does backward receive z (pre-activation)?**
The chain rule defines the activation derivative in terms of z. Sigmoid's derivative recomputes sigmoid internally, but the cost is negligible. Accepting z keeps the backward interface agnostic to activation type (single responsibility).

**Why separate Model and Network classes?**
Model owns the training loop, data flow, and serialization. Network owns forward and backward computation. Splitting responsibilities keeps each class focused and testable independently.

## Default Configuration

- Hidden layers: `[24, 24, 24]`
- Hidden activation: ReLU (`--activation relu`)
- Weight initialization: He Uniform (`--weights_initializer heUniform`)
- Output: 2 neurons (binary classification via softmax)
- Loss: Cross-entropy (only supported loss; unsupported combinations raise an error)
- Optimizer: Adam (lr=0.0001)
- Batch size: 4, Epochs: 200
- Early stopping patience: 10 epochs
- Random seed: 42

## Installation

1. Set up the virtual environment:
   ```bash
   python -m .venv venv
   source .venv/bin/activate
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
python main.py --help
```

Example:
```bash
python main.py --data train.csv --val_data validation.csv --layers 24 24 24 --epochs 200 --batch_size 4 --learning_rate 0.0001 --solver adam --activation relu --weights_initializer heUniform --seed 42
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to the training data CSV file | `train.csv` |
| `--val_data` | Path to the validation data CSV file | `validation.csv` |
| `--layers` | Sizes of hidden layers | `24 24 24` |
| `--epochs` | Number of training epochs | `200` |
| `--batch_size` | Batch size for training | `4` |
| `--learning_rate` | Learning rate | `0.0001` |
| `--solver` | Optimizer (`adam` or `sgd`) | `adam` |
| `--activation` | Hidden activation: `relu` or `sigmoid` | `relu` |
| `--weights_initializer` | Weight initialization: `heUniform` or `xavierUniform` | `heUniform` |
| `--seed` | Random seed for reproducibility | `42` |

The training script saves the trained model to `model.pkl` and generates a `learning_curve.png` plot.

### Prediction

```bash
python predict.py --data validation.csv --model model.pkl
```

| Argument | Description | Default |
|---|---|---|
| `--data` | Path to the input data CSV file | `validation.csv` |
| `--model` | Path to the trained model file | `model.pkl` |
