"""K. I/O tests."""
import numpy as np
import pytest
import pickle
import os
import model as model_module


def test_save_load_predictions_match(tmp_path):
    """K1/K2: Predictions identical after save-load cycle."""
    np.random.seed(42)
    mlp = model_module.Model(
        hidden_layer_sizes=[24, 24, 24],
        learning_rate=0.001, epochs=20, batch_size=8,
        solver="adam", output_activation="softmax",
        loss="cross_entropy", weights_initializer="heUniform",
    )
    x = np.random.randn(50, 30)
    y = np.zeros((50, 2))
    y[np.arange(50), np.random.randint(0, 2, 50)] = 1
    mlp.fit(x[:40], y[:40])

    x_test = (x[40:] - mlp.mean_train) / mlp.std_train
    pred_before = mlp.predict(x_test)

    path = str(tmp_path / "model.pkl")
    mlp.save(path)

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    pred_after = loaded.predict((x[40:] - loaded.mean_train) / loaded.std_train)
    np.testing.assert_array_almost_equal(pred_before, pred_after)


@pytest.mark.parametrize("arg", [
    "--data", "--layers", "--epochs", "--batch_size",
    "--learning_rate", "--split", "--solver", "--seed",
])
def test_argparse_has_param(arg):
    """K3: main.py exposes required CLI arguments."""
    main_path = os.path.join(os.path.dirname(__file__), "..", "main.py")
    with open(main_path) as f:
        src = f.read()
    assert arg in src


def test_seed_reproducibility():
    """K4: Same seed -> identical final loss."""
    def run(seed):
        np.random.seed(seed)
        mlp = model_module.Model(
            hidden_layer_sizes=[24, 24, 24],
            learning_rate=0.001, epochs=10, batch_size=8,
            solver="adam", output_activation="softmax",
            loss="cross_entropy", weights_initializer="heUniform",
        )
        x = np.random.randn(30, 30)
        y = np.zeros((30, 2))
        y[np.arange(30), np.random.randint(0, 2, 30)] = 1
        h = mlp.fit(x, y)
        return h["loss"][-1]

    assert run(42) == run(42)
