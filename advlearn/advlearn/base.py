"""Base classes."""

import numpy as np
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize, OptimizeResult
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod


class BaseAttack(BaseEstimator, metaclass=ABCMeta):
    """Base class for all attack methods."""

    @abstractmethod
    def attack(self, n_points):
        """Generate attack points

        Parameters
        ----------
        n_points : int
            Number of points

        Returns
        -------
        x_c : ndarray
            Attack points
        """
        pass

    @abstractmethod
    def surrogate_model(self):
        """Get the surrogate model

        Returns
        -------
        model : sklearn model
            Surrogate model the attacker targets
        """
        pass


class OptimizableAttackMixin(object, metaclass=ABCMeta):
    """Mixin for all attacks using gradient optimization methods"""

    def __init__(self, opt_method='GD', stepsize=0.1, n_steps=100, atol=1e-3):
        self.opt_method = opt_method
        self.stepsize = stepsize
        self.n_steps = n_steps
        self.atol = atol

    @abstractmethod
    def attack_direction(self, x_c, y_c):
        pass

    @abstractmethod
    def attack_loss(self, x_c, y_c):
        pass

    def attack_trajectory(self, attack_data, attack_label, extra_data=None, extra_labels=None):
        result = self._optimize(attack_data, attack_label,
                                extra_data=extra_data, extra_labels=extra_labels,
                                trajectory=True)
        return result.history

    def attack_optimize(self, attack_data, attack_labels):
        """Determine the optimal attack points from a set of initial points

        Parameters
        ----------
        attack_data
        attack_labels
        opt_method
        trajectory

        Returns
        -------

        """
        opt_attack_data = np.zeros_like(attack_data)

        n_points = attack_data.shape[0]
        for ind in range(n_points):
            ind_except = np.arange(n_points)

            attack_data_ind = attack_data[ind, :]
            attack_label_ind = attack_labels[ind]
            extra_data = attack_data[ind_except != ind, :]
            extra_labels = attack_labels[ind_except != ind]

            result = self._optimize(attack_data_ind, attack_label_ind, extra_data, extra_labels)
            opt_attack_data[ind, :] = result.x

        return opt_attack_data

    def _optimize(self, attack_data, attack_label, extra_data, extra_labels, trajectory=False):
        """Optimize an attack point using scipy optimize.

        Parameters
        ----------
        attack_data
        attack_labels
        extra_data
        extra_labels
        opt_method
        trajectory

        Returns
        -------
        result
        """

        # As we would like to maximize instead of minimize the function
        def wrap_jac(jac_attack_data, jac_attack_label, jac_extra_data, jac_extra_labels):
            return - self.attack_direction(jac_attack_data, jac_attack_label,
                                           extra_data=jac_extra_data,
                                           extra_labels=jac_extra_labels).flatten()

        def wrap_fun(fun_attack_data, fun_attack_label, fun_extra_data, fun_extra_labels):
            return - self.attack_loss(fun_attack_data, fun_attack_label,
                                      extra_data=fun_extra_data,
                                      extra_labels=fun_extra_labels)

        # Call the scipy optimization routine
        if self.opt_method == 'GD':
            result = minimize(wrap_fun, attack_data, method=self._minimize_gd,
                              args=(attack_label, extra_data, extra_labels),
                              jac=wrap_jac, options={'stepsize': self.stepsize, 'maxiter': self.n_steps,
                                                     'atol': self.atol, 'retall': trajectory})
        elif self.opt_method in ['CG', 'Newton-CG', 'Nelder-Mead']:
            result = minimize(wrap_fun, attack_data, method=self.opt_method,
                              args=(attack_label, extra_data, extra_labels),
                              jac=wrap_jac)
        else:
            raise ValueError('Invalid optimization method')

        return result

    def _minimize_gd(self, fun, x0, args=(), atol=1e-3, stepsize=0.3, jac=None,
                     retall=False, maxiter=1000, **kwargs):
        """ Scipy custom gradient descent optimizer

        Parameters
        ----------
        fun : callable
            Function to optimize
        x0 : ndarray
            Initial position
        args : tuple
            Arguments to pass to fun, jac
        atol : float
            Absolute tolerance termination condition
        stepsize : float
            Optimization step size
        jac : callable
            Gradient or jacobian matrix
        retall : bool
            Store data after each step?
        maxiter : int
            Maximum number of iterations
        kwargs
            Additional keyword arguments

        Returns
        -------
        results : OptimizeResult
            Results of optimization
        """
        if retall:
            history = deepcopy(x0).reshape(1, -1)

        x_current = x0
        x_prev = deepcopy(x0)

        for step in range(maxiter):
            x_current = x_current - stepsize * jac(x_current, *args)

            # Save the history
            if retall:
                history = np.append(history, deepcopy(x_current).reshape(1, -1), axis=0)

            # Absolute tolerance convergence
            if np.allclose(x_current, x_prev, rtol=0, atol=atol):
                break

            x_prev = deepcopy(x_current)

        res = OptimizeResult()
        res.x = x_current
        if retall:
            res.history = history
        return res


class IntelligentAttackMixin(object):
    def __init__(self, outlier_method=None, outlier_weight=1,
                 outlier_distance_threshold=1, outlier_power=2,
                 outlier_k=3):
        self.outlier_method = outlier_method
        self.outlier_weight = outlier_weight
        self.outlier_distance_threshold = outlier_distance_threshold
        self.outlier_power = outlier_power
        self.outlier_k = outlier_k

    def _intelligent_loss(self, loss, attack_data, attack_label):
        """

        Parameters
        ----------
        loss
        attack_data
        attack_label

        Returns
        -------

        """

        if self.outlier_method is None:
            loss_updated = loss
        elif self.outlier_method == 'distancethreshold':
            nbrs = NearestNeighbors(n_neighbors=1).fit(self.data)
            dists, _ = nbrs.kneighbors(attack_data.reshape(1, -1))
            if dists[0, -1] > self.outlier_distance_threshold:
                loss_updated = - np.inf
            else:
                loss_updated = loss
        elif self.outlier_method == 'kthdistance':
            nbrs = NearestNeighbors(n_neighbors=self.outlier_k).fit(self.data)
            dists, _ = nbrs.kneighbors(attack_data.reshape(1, -1))
            loss_updated = loss - self.outlier_weight * (dists[0, -1] ** self.outlier_power)
        else:
            raise ValueError('Invalid outlier method')

        return loss_updated

    def _intelligent_direction(self, direction, attack_data, attack_label):
        """

        Parameters
        ----------
        direction
        attack_data
        attack_label

        Returns
        -------

        """

        if self.outlier_method is None:
            direction_updated = direction
        elif self.outlier_method == 'distancethreshold':
            neighbors = NearestNeighbors(n_neighbors=1).fit(self.data)
            dists, _ = neighbors.kneighbors(attack_data.reshape(1, -1))
            if dists[0, -1] < self.outlier_distance_threshold:
                direction_updated = direction
            else:
                direction_updated = np.zeros_like(direction)
        elif self.outlier_method == 'kthdistance':
            neighbors = NearestNeighbors(
                n_neighbors=self.outlier_k).fit(self.data)
            _, idx = neighbors.kneighbors(attack_data.reshape(1, -1))
            norm = np.linalg.norm((attack_data - self.data[idx[0, -1]]), ord=2)
            grad_penalty = self.outlier_power * \
                           (norm ** (self.outlier_power - 2)) * \
                           (attack_data - self.data[idx[0, -1]])
            direction_updated = direction - self.outlier_weight * grad_penalty
        else:
            raise ValueError('Invalid outlier method')

        return direction_updated


class PoisonMixin(object):
    """Mixin for all poisoning attacks.
    """
    pass


class EvadeMixin(object):
    """Mixin for all evasion attacks.
    """
    pass


class BaseDefense(BaseEstimator):
    """Base class for all defenses.
    """
    pass
