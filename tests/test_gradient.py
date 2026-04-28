"""A. Gradient Check — numerical vs analytical gradient."""
import numpy as np
import pytest
from network import Network, NetworkConfig


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
@pytest.mark.parametrize("param_type", ["weight", "bias"])
def test_relu_softmax_ce(
    small_network, small_data, numerical_gradient, relative_error,
    layer_idx, param_type,
):
    """A1/A2: Per-layer gradient check (ReLU + Softmax + CE)."""
    x, y = small_data
    small_network.forward(x)
    nabla_w, nabla_b = small_network.backward(y)

    analytical = nabla_w[layer_idx] if param_type == "weight" else nabla_b[layer_idx]
    numerical = numerical_gradient(small_network, x, y, param_type, layer_idx)
    assert relative_error(numerical, analytical) < 1e-5


@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("param_type", ["weight", "bias"])
def test_sigmoid_softmax_ce(
    numerical_gradient, relative_error, layer_idx, param_type,
):
    """A3: Gradient check with sigmoid hidden + Softmax + CE."""
    np.random.seed(123)
    net = Network(NetworkConfig(
        layers=[4, 3, 2], activation="sigmoid", loss="cross_entropy",
        output_activation="softmax", weights_initializer="xavierUniform",
    ))
    x = np.random.randn(3, 4) * 0.5
    y = np.array([[1, 0], [0, 1], [1, 0]])

    net.forward(x)
    nabla_w, nabla_b = net.backward(y)

    analytical = nabla_w[layer_idx] if param_type == "weight" else nabla_b[layer_idx]
    numerical = numerical_gradient(net, x, y, param_type, layer_idx)
    assert relative_error(numerical, analytical) < 1e-5
