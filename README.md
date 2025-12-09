# Multilayer Perceptron (MLP) Project

This project implements a Multilayer Perceptron (MLP) from scratch using Python and NumPy. It is designed to classify data, specifically tailored for the Breast Cancer Wisconsin (Diagnostic) Data Set, distinguishing between Malignant (M) and Benign (B) tumors.

## Basics of Multilayer Perceptron (MLP)

A Multilayer Perceptron (MLP) is a class of feedforward artificial neural networks (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

### Key Concepts:
- **Input Layer**: Receives the initial data.
- **Hidden Layers**: Intermediate layers where the computation and feature extraction happen. Deep learning models have multiple hidden layers.
- **Output Layer**: Produces the final prediction or classification.
- **Weights and Biases**: Parameters that the model learns during training.
- **Activation Function**: Introduces non-linearity to the model (e.g., Sigmoid, ReLU, Softmax).
- **Forward Propagation**: The process of passing the input through the network to generate an output.
- **Backpropagation**: The algorithm used to calculate the gradient of the loss function with respect to the weights, allowing the model to learn by updating weights to minimize error.

## Project Structure

- `multilayer_perceptron.py`: Main class interface for the MLP.
- `model.py`: Handles the training loop and model management.
- `network.py`: Defines the neural network structure, forward/backward propagation.
- `optimizer.py`: Implements optimization algorithms (SGD, Adam).
- `utils.py`: Contains utility functions (activation functions, loss functions).
- `main_train.py`: Script to train the model.
- `predict.py`: Script to load a trained model and make predictions.
- `data.csv`: Dataset file (must be present for training).

## Installation

1. **Clone the repository** (if applicable).
2. **Set up the Virtual Environment**:
   Run the provided script to create a virtual environment and install dependencies.
   ```bash
   ./start_venv.sh
   source venv/bin/activate
   ```
   Alternatively, you can manually install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, use `main_train.py`. You can inspect the available arguments using `--help`.

```bash
python main_train.py --help
```

**Example command:**
```bash
python main_train.py --data data.csv --layers 24 24 24 --epochs 200 --batch_size 4 --learning_rate 0.0001 --split 0.2 --solver adam
```

- `--data`: Path to the training data CSV file (default: `data.csv`).
- `--layers`: Sizes of hidden layers (default: `24 24 24`).
- `--epochs`: Number of training epochs (default: `200`).
- `--batch_size`: Batch size for training (default: `4`).
- `--learning_rate`: Learning rate (default: `0.0001`).
- `--split`: Train/validation split ratio (default: `0.2`).
- `--solver`: Select optimizer adam or sgd (default: `adam`).

The training script will save the trained model to `model.pkl` and generate a learning curve plot `learning_curve.png`.

### Prediction

To make predictions using a trained model, use `predict.py`.

```bash
python predict.py --data data.csv --model model.pkl
```

- `--data`: Path to the input data CSV file for prediction (default: `data.csv`).
- `--model`: Path to the trained model file (default: `model.pkl`).

This will output the predicted labels (e.g., `['M', 'B', ...]`).
