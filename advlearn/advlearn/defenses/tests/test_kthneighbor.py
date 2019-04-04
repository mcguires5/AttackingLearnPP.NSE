"""Test dummy defense"""

import numpy as np
from advlearn.defenses import KthNeighbor

X = np.array(
    [
        [-0.12840393, 0.66446571],
        [1.32319756, -0.13181616],
        [0.04296502, -0.37981873],
        [0.83631853, 0.18569783],
    ]
)
Y = np.array([1, 2, 2, 2])


class TestDummyDefense(object):
    """Test dummy defense"""

    def test_defense_transform_x(self):
        """Test the output given only data"""
        model = KthNeighbor()
        X_out, used = model.transform(X)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(used, np.ndarray)
        assert used.ndim == 1
        assert np.array_equal(X_out, X[used, :])

    def test_defense_transform_x_y(self):
        """Test that the output given both data and labels"""
        model = KthNeighbor()
        X_out, Y_out, used = model.transform(X, Y)
        assert isinstance(X_out, np.ndarray)
        assert isinstance(Y_out, np.ndarray)
        assert isinstance(used, np.ndarray)
        assert used.ndim == 1
        assert np.array_equal(X_out, X[used, :])
        assert np.array_equal(Y_out, Y[used])
