"""H. Activation  &  G. Weight Initialization tests."""
import numpy as np
import pytest
from utils import relu, relu_prime, sigmoid, softmax
from network import Network, NetworkConfig


# ── H1: ReLU ─────────────────────────────────────────────────

class TestReLU:
    def test_forward(self):
        """max(0, x)."""
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        np.testing.assert_array_equal(relu(x), [0, 0, 0, 1, 2])

    def test_derivative(self):
        """1 if x > 0 else 0."""
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        np.testing.assert_array_equal(relu_prime(x), [0, 0, 0, 1, 1])


# ── H2: Sigmoid stability ────────────────────────────────────

@pytest.mark.parametrize("val", [-1000, -100, 0, 100, 1000])
def test_sigmoid_in_range(val):
    """Output in [0, 1] for extreme inputs, no NaN."""
    result = sigmoid(np.array([[val]]))[0, 0]
    assert 0 <= result <= 1
    assert not np.isnan(result)


# ── H3: Softmax only on output ───────────────────────────────

def test_softmax_output_only(small_network):
    """Output sums to 1; hidden layers do not."""
    x = np.random.randn(3, 4)
    output = small_network.forward(x)

    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-6)

    for act in small_network.activations[1:-1]:
        assert not np.allclose(act.sum(axis=1), 1.0)


# ── H4: Dead ReLU ────────────────────────────────────────────

def test_dead_relu_below_50pct(small_network):
    """No hidden layer has > 50 % dead neurons."""
    x = np.random.randn(200, 4)
    small_network.forward(x)
    for act in small_network.activations[1:-1]:
        dead_pct = np.all(act == 0, axis=0).sum() / act.shape[1]
        assert dead_pct < 0.5


# ── G1: No all-zero init ─────────────────────────────────────

def test_no_zero_init():
    np.random.seed(42)
    net = Network(NetworkConfig(
        layers=[30, 24, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))
    for w in net.weights:
        assert not np.allclose(w, 0)


# ── G2: He / Xavier range ────────────────────────────────────

def test_he_uniform_range():
    """Weights within +/- sqrt(6 / fan_in)."""
    np.random.seed(42)
    net = Network(NetworkConfig(
        layers=[100, 50, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))
    limit = np.sqrt(6 / 100)
    assert np.all(np.abs(net.weights[0]) <= limit + 1e-10)


def test_xavier_uniform_range():
    """Weights within +/- sqrt(6 / (fan_in + fan_out))."""
    np.random.seed(42)
    net = Network(NetworkConfig(
        layers=[100, 50, 2], activation="sigmoid", loss="cross_entropy",
        output_activation="softmax", weights_initializer="xavierUniform",
    ))
    limit = np.sqrt(6 / (100 + 50))
    assert np.all(np.abs(net.weights[0]) <= limit + 1e-10)


# ── G3: Bias zero ────────────────────────────────────────────

def test_bias_zero_init():
    np.random.seed(42)
    net = Network(NetworkConfig(
        layers=[30, 24, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))
    for b in net.biases:
        np.testing.assert_array_equal(b, 0)
