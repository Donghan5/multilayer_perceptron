"""C. Numerical Stability tests (C1, C2, C3)."""
import numpy as np
import pytest
from utils import softmax, cross_entropy
from network import Network, NetworkConfig
from optimizer import Adam


# ── C1: Softmax extreme ──────────────────────────────────────

@pytest.mark.parametrize("x", [
    np.array([[1000, 2000, 3000]]),
    np.array([[-1000, -2000, -3000]]),
    np.array([[1e10, 1e10, 1e10]]),
], ids=["large_pos", "large_neg", "equal_huge"])
def test_softmax_extreme(x):
    """No NaN/Inf, sums to 1."""
    result = softmax(x)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


# ── C2: Cross-entropy boundary ───────────────────────────────

@pytest.mark.parametrize("y_pred", [
    np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
    np.array([[0.999999, 1e-15], [1e-15, 0.999999], [0.999999, 1e-15]]),
    np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
], ids=["normal", "near_zero", "exact_boundary"])
def test_cross_entropy_boundary(y_pred):
    """No NaN/Inf for boundary predictions."""
    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    loss = cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


# ── C3: NaN/Inf monitoring during training ────────────────────

def test_no_nan_during_training(synthetic_data):
    """200-epoch training produces no NaN/Inf in loss, grads, or weights."""
    x, y = synthetic_data
    net = Network(NetworkConfig(
        layers=[30, 24, 24, 24, 2], activation="relu", loss="cross_entropy",
        output_activation="softmax", weights_initializer="heUniform",
    ))
    opt = Adam(learning_rate=0.001)

    for epoch in range(200):
        out = net.forward(x)
        loss = net.loss(y, out)
        assert not np.isnan(loss), f"NaN loss at epoch {epoch}"
        assert not np.isinf(loss), f"Inf loss at epoch {epoch}"

        nw, nb = net.backward(y)
        for i, (gw, gb) in enumerate(zip(nw, nb)):
            assert not np.any(np.isnan(gw)), f"NaN grad_w layer {i} epoch {epoch}"
            assert not np.any(np.isnan(gb)), f"NaN grad_b layer {i} epoch {epoch}"

        opt.update(net, nw, nb)

        for i, (w, b) in enumerate(zip(net.weights, net.biases)):
            assert not np.any(np.isnan(w)), f"NaN weight layer {i} epoch {epoch}"
            assert not np.any(np.isnan(b)), f"NaN bias layer {i} epoch {epoch}"
