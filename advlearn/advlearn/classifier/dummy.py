"""Dummy classifiers for testing."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier for testing
    """

    def __init__(self):
        self.unique_y = 0

    def fit(self, X, y):  # pylint: disable=unused-argument
        """Fit the dummy classifier

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        """
        self.unique_y = np.unique(y)

    def partial_fit(self, X, y):  # pylint: disable=unused-argument
        """Update the dummy classifier

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        """
        self.unique_y = np.unique(np.concatenate((y, self.unique_y)))

    def predict(self, X):
        """Predict with the dummy classifier

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        y_out : np.ndarray
        """
        return np.random.choice(self.unique_y, size=(X.shape[0]))
