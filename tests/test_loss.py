"""I. Loss function tests."""
import numpy as np
import pytest
from utils import cross_entropy, one_hot_encode


def test_cross_entropy_correctness():
    """I1: Code matches manual computation."""
    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    eps = 1e-15
    manual = -np.sum(y_true * np.log(np.clip(y_pred, eps, 1 - eps))) / y_true.shape[0]
    assert np.isclose(cross_entropy(y_true, y_pred), manual)


def test_batch_average_not_sum():
    """I2: Loss is mean over batch, not sum."""
    y1 = np.array([[1, 0]])
    p1 = np.array([[0.9, 0.1]])
    y5 = np.tile(y1, (5, 1))
    p5 = np.tile(p1, (5, 1))
    assert np.isclose(cross_entropy(y1, p1), cross_entropy(y5, p5))


@pytest.mark.parametrize("label,expected", [("M", [0, 1]), ("B", [1, 0])])
def test_one_hot_encode(label, expected):
    """I3: M -> [0,1], B -> [1,0]."""
    encoded = one_hot_encode(np.array([label]))
    np.testing.assert_array_equal(encoded[0], expected)


def test_one_hot_row_sums():
    """I3: Each row sums to 1."""
    encoded = one_hot_encode(np.array(["M", "B", "M", "B", "M"]))
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(5))
