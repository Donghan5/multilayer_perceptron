"""L. Edge Case tests."""
import numpy as np
import pytest
from network import Network, NetworkConfig
from optimizer import Adam


def _make(input_dim):
    """Create a small network + Adam optimizer."""
    net = Network(NetworkConfig(
        layers=[input_dim, 8, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))
    return net, Adam(learning_rate=0.001)


def test_all_same_class():
    """L1: All-same-class data produces no NaN."""
    net, opt = _make(10)
    x = np.random.randn(50, 10)
    y = np.zeros((50, 2))
    y[:, 0] = 1

    for _ in range(100):
        net.forward(x)
        nw, nb = net.backward(y)
        opt.update(net, nw, nb)

    assert not np.any(np.isnan(net.forward(x)))


def test_nan_propagation():
    """L2: NaN in input propagates (no silent corruption)."""
    net, _ = _make(10)
    x = np.random.randn(5, 10)
    x[0, 0] = np.nan
    assert np.any(np.isnan(net.forward(x)))


def test_different_scales_normalized():
    """L3: Normalization handles wildly different feature scales."""
    x = np.column_stack([
        np.random.randn(50) * 1e6,
        np.random.randn(50) * 1e-6,
        np.random.randn(50) * 100,
        np.ones(50) * 42,
    ])
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x_norm = (x - mean) / std

    assert not np.any(np.isnan(x_norm))
    assert not np.any(np.isinf(x_norm))

    net, _ = _make(4)
    assert not np.any(np.isnan(net.forward(x_norm)))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_tiny_dataset(n):
    """L4: Training doesn't crash with n < 10 samples."""
    net, opt = _make(5)
    x = np.random.randn(n, 5)
    y = np.zeros((n, 2))
    y[np.arange(n), np.random.randint(0, 2, n)] = 1

    for _ in range(50):
        net.forward(x)
        nw, nb = net.backward(y)
        opt.update(net, nw, nb)

    assert not np.isnan(net.loss(y, net.forward(x)))
