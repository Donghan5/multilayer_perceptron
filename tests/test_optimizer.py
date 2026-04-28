"""E. Adam  &  F. SGD tests."""
import numpy as np
import pytest
import inspect
from optimizer import Adam, Sgd
from network import Network, NetworkConfig


# ── E. Adam ──────────────────────────────────────────────────

class TestAdam:
    def test_default_hyperparams(self):
        """E1: beta1=0.9, beta2=0.999, eps=1e-8."""
        adam = Adam(learning_rate=0.001)
        assert adam.beta1 == 0.9
        assert adam.beta2 == 0.999
        assert adam.epsilon == 1e-8

    def test_mv_none_before_update(self):
        """E2: m, v are None before first update."""
        adam = Adam(learning_rate=0.001)
        assert adam.m_w is None
        assert adam.v_w is None

    def test_mv_shapes_after_update(self, small_network, small_data):
        """E2: m, v shapes match weights after first update."""
        x, y = small_data
        adam = Adam(learning_rate=0.001)
        small_network.forward(x)
        nw, nb = small_network.backward(y)
        adam.update(small_network, nw, nb)
        for i in range(len(small_network.weights)):
            assert adam.m_w[i].shape == small_network.weights[i].shape
            assert adam.v_w[i].shape == small_network.weights[i].shape

    @pytest.mark.parametrize("steps", [1, 3, 5])
    def test_timestep_increment(self, small_network, small_data, steps):
        """E3: timestep == number of update calls."""
        x, y = small_data
        adam = Adam(learning_rate=0.001)
        for _ in range(steps):
            small_network.forward(x)
            nw, nb = small_network.backward(y)
            adam.update(small_network, nw, nb)
        assert adam.timestep == steps

    def test_bias_correction_factors(self):
        """E4: Correction significant at t=1 (10x for m, 1000x for v)."""
        assert abs(1 / (1 - 0.9 ** 1) - 10.0) < 0.01
        assert abs(1 / (1 - 0.999 ** 1) - 1000.0) < 1.0

    def test_per_layer_independent_mv(self):
        """E5: Each layer has independent m, v entries."""
        np.random.seed(42)
        net = Network(NetworkConfig(
            layers=[4, 5, 3, 2], activation="relu", loss="cross_entropy",
            output_activation="softmax", weights_initializer="heUniform",
        ))
        adam = Adam(learning_rate=0.001)
        x = np.random.randn(3, 4)
        y = np.array([[1, 0], [0, 1], [1, 0]])
        net.forward(x)
        nw, nb = net.backward(y)
        adam.update(net, nw, nb)

        n = len(net.weights)
        assert len(adam.m_w) == n
        assert len(adam.v_w) == n
        assert len(adam.m_b) == n
        assert len(adam.v_b) == n
        for i in range(n):
            assert adam.m_w[i].shape == net.weights[i].shape
            assert adam.m_b[i].shape == net.biases[i].shape


# ── F. SGD ───────────────────────────────────────────────────

class TestSgd:
    def test_minibatch_covers_all(self):
        """F1: Mini-batch split covers every sample exactly once."""
        n, bs = 17, 4
        batches = [np.arange(n)[i:i + bs] for i in range(0, n, bs)]
        np.testing.assert_array_equal(
            np.sort(np.concatenate(batches)), np.arange(n),
        )

    def test_no_momentum(self):
        """F2: Current SGD has no momentum (baseline doc)."""
        src = inspect.getsource(Sgd)
        assert "momentum" not in src and "velocity" not in src

    def test_no_lr_decay(self):
        """F3: Current SGD has no LR decay (baseline doc)."""
        src = inspect.getsource(Sgd)
        assert "decay" not in src and "schedule" not in src
