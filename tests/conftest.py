"""Shared fixtures and helpers for MLP test suite."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network import Network, NetworkConfig
from utils import one_hot_encode
import model as model_module


# ── Helper fixtures (return callables) ───────────────────────

@pytest.fixture
def numerical_gradient():
    """Central-difference numerical gradient."""
    def _compute(network, x, y, param_type, layer_idx, epsilon=1e-5):
        params = (network.weights[layer_idx]
                  if param_type == "weight"
                  else network.biases[layer_idx])
        grad = np.zeros_like(params)
        it = np.nditer(params, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = float(params[idx])
            params[idx] = old + epsilon
            loss_p = network.loss(y, network.forward(x))
            params[idx] = old - epsilon
            loss_m = network.loss(y, network.forward(x))
            params[idx] = old
            grad[idx] = (loss_p - loss_m) / (2 * epsilon)
            it.iternext()
        return grad
    return _compute


@pytest.fixture
def relative_error():
    """Max relative error between two arrays."""
    def _compute(numerical, analytical):
        denom = np.maximum(np.abs(numerical) + np.abs(analytical), 1e-8)
        return np.max(np.abs(numerical - analytical) / denom)
    return _compute


@pytest.fixture
def compute_metrics():
    """Precision, recall, F1 for a single class."""
    def _compute(y_true, y_pred, class_idx):
        tp = np.sum((y_pred == class_idx) & (y_true == class_idx))
        fp = np.sum((y_pred == class_idx) & (y_true != class_idx))
        fn = np.sum((y_pred != class_idx) & (y_true == class_idx))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1
    return _compute


# ── Data / network fixtures ──────────────────────────────────

@pytest.fixture(autouse=True)
def fixed_seed():
    """Reset RNG before each test."""
    np.random.seed(42)


@pytest.fixture
def small_network():
    """4->5->3->2 ReLU + Softmax + CE."""
    return Network(NetworkConfig(
        layers=[4, 5, 3, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))


@pytest.fixture
def small_data():
    """5 samples, 4 features, 2 classes."""
    x = np.random.randn(5, 4)
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    return x, y


@pytest.fixture
def synthetic_data():
    """100 samples, 30 features, 2 classes."""
    x = np.random.randn(100, 30)
    y = np.zeros((100, 2))
    y[np.arange(100), np.random.randint(0, 2, 100)] = 1
    return x, y


@pytest.fixture(scope="session")
def trained_model():
    """Train on data.csv, cached for the session. Skip if unavailable."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data.csv")
    if not os.path.exists(data_path):
        pytest.skip("data.csv not found")

    import pandas as pd
    np.random.seed(42)

    df = pd.read_csv(data_path, header=None)
    y_raw = df.iloc[:, 1].values
    X_raw = df.iloc[:, 2:].values
    y = one_hot_encode(y_raw)

    indices = np.arange(len(X_raw))
    np.random.shuffle(indices)
    split = int(len(X_raw) * 0.8)

    X_train, X_val = X_raw[indices[:split]], X_raw[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]

    mlp = model_module.Model(
        hidden_layer_sizes=[24, 24, 24],
        learning_rate=0.003, epochs=200, batch_size=8,
        solver="adam", activation="sigmoid",
        output_activation="softmax",
        loss="cross_entropy", weights_initializer="heUniform",
    )
    history = mlp.fit(X_train, y_train, X_val, y_val)

    X_val_norm = (X_val - mlp.mean_train) / mlp.std_train
    val_out = mlp.predict(X_val_norm)

    return {
        "model": mlp,
        "history": history,
        "y_pred": np.argmax(val_out, axis=1),
        "y_true": np.argmax(y_val, axis=1),
    }
