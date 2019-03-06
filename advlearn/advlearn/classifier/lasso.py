"""Online lasso regression"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Lasso


class OnlineLasso(BaseEstimator, ClassifierMixin):
    """
    Any classification algorithms that are used with the ensemble package
    must implement this interface
    """

    def __init__(self, alpha=0.1):
        self.x = None
        self.y = None
        self.lasso = Lasso(alpha=alpha, fit_intercept=True)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        """
        :param x: (np.array) next batch of data to train on
        :param y: (np.array) labels that correspond to batch
        """
        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = np.append(self.x, x, axis=0)
            self.y = np.append(self.y, y)

        self.lasso.fit(self.x, self.y)

    def partial_fit(self, x, y):
        """Update the classifier

        Parameters
        ----------
        x : np.ndarray
        y : np.ndarray
        """
        self.fit(x, y)

    def predict(self, x):
        """
        :param x: (np.array) next batch of data to train on
        :return y: (np.array) labels that the classifier gives to the data
        """
        return np.sign(self.lasso.predict(x))
