"""Dummy attacks for testing."""

import numpy as np
from advlearn.base import BaseAttack, PoisonMixin, EvadeMixin


class DummyPoisonAttack(BaseAttack, PoisonMixin):
    """Dummy poisoning attack"""

    def __init__(self):
        self.n_dim = 0

    def fit(self, X, y):  # pylint: disable=unused-argument
        """Fit dummy poisoning attack

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        """
        self.n_dim = X.shape[1]

    def get_attack_point(self):
        """Generate attack point from dummy poisoning attack

        Returns
        -------
        X : np.ndarray
        y : np.ndarray
        """
        return np.random.rand(1, self.n_dim), np.array([1])


class DummyEvadeAttack(BaseAttack, EvadeMixin):
    """Dummy evasion attack"""

    def __init__(self):
        self.n_dim = 0

    def fit(self, X, y):  # pylint: disable=unused-argument
        """Fit dummy evasion attack

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        """
        self.n_dim = X.shape[1]

    def get_attack_point(self):
        """Generate attack point from dummy evasion attack

        Returns
        -------
        y : np.ndarray
        """
        return np.random.rand(1, self.n_dim), 1
