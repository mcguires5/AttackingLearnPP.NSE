"""Dummy defenses for testing."""

import numpy as np
from sklearn.base import BaseEstimator


class DummyDefense(BaseEstimator):
    """Dummy defense for testing

    Defenses can alter the input data in many ways.
    It can add and remove data. It can also change the features.
    This dummy defense removes random data from the input.
    """

    def transform(self, X, y=None):  # pylint: disable=no-self-use
        """Transform the input data

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        X_new : np.ndarray
        y_new : np.ndarray
        used : np.ndarray

        """
        mask = np.random.choice([False, True], X.shape[0], p=[0.75, 0.25])
        used = mask.nonzero()[0]
        if y is None:
            return X[mask, :], used
        else:
            return X[mask, :], y[mask], used

    def fit(self, X, y):
        """Fit the model on the data

        In most functions this is important, but not here.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        """
        pass

    def fit_transform(self, X, y):
        """Fit the model and then transform the input

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        X_new : np.ndarray
        y_new : np.ndarray
        used : np.ndarray
        """
        self.fit(X, y)
        return self.transform(X, y)
