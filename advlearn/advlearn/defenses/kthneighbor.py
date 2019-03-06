"""Kth Neighbor defense for testing."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


class KthNeighbor(BaseEstimator):
    """Kth Neighbor defense for testing
    """

    def __init__(self, outlier_distance_threshold=1):
        self.outlier_distance_threshold = outlier_distance_threshold
        self.X = None

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
        # Normally only compare the new data to what it has seen in the past, but this doesn't work the first time
        if self.X is None:
            self.X = X

        nbrs = NearestNeighbors(n_neighbors=2).fit(self.X)
        dists, _ = nbrs.kneighbors(X)
        mask = dists[:,-1] < self.outlier_distance_threshold

        self.X = np.vstack((self.X, X[mask, :]))

        if y is None:
            return X[mask, :], mask.nonzero()[0]
        else:
            return X[mask, :], y[mask], mask.nonzero()[0]

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
