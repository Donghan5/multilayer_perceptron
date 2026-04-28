"""D. Training Loop tests (D1, D2, D4)."""
import numpy as np
from network import Network, NetworkConfig
from optimizer import Adam


def test_shuffle_different_each_epoch():
    """D1: Shuffled indices differ across 5 epochs."""
    np.random.seed(42)
    orders = []
    for _ in range(5):
        idx = np.arange(100)
        np.random.shuffle(idx)
        orders.append(idx.copy())

    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            assert not np.array_equal(orders[i], orders[j])


def test_standardization_train_fit_val_transform():
    """D2: Normalise with train stats; val mean != 0."""
    np.random.seed(42)
    X = np.random.randn(100, 5) * 10 + 50
    X_tr, X_val = X[:80], X[80:]

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr_n = (X_tr - mean) / std
    X_val_n = (X_val - mean) / std

    np.testing.assert_allclose(X_tr_n.mean(axis=0), 0, atol=0.15)
    np.testing.assert_allclose(X_tr_n.std(axis=0), 1, atol=0.15)
    assert not np.allclose(X_val_n.mean(axis=0), 0, atol=0.01)


def test_seed_deterministic_training():
    """D4: Same seed -> identical final loss."""
    def run(seed):
        np.random.seed(seed)
        net = Network(NetworkConfig(
            layers=[30, 24, 24, 24, 2], activation="relu",
            loss="cross_entropy", output_activation="softmax",
            weights_initializer="heUniform",
        ))
        opt = Adam(learning_rate=0.001)
        x = np.random.randn(50, 30)
        y = np.zeros((50, 2))
        y[np.arange(50), np.random.randint(0, 2, 50)] = 1
        for _ in range(50):
            net.forward(x)
            nw, nb = net.backward(y)
            opt.update(net, nw, nb)
        return net.loss(y, net.forward(x))

    assert run(42) == run(42)
