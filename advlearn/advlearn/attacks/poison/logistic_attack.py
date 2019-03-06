"""General Logistic Poison Attack"""

import numpy as np
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from advlearn.base import BaseAttack, PoisonMixin, OptimizableAttackMixin
from advlearn.utils import Projector


class LogisticAttack(BaseAttack, OptimizableAttackMixin, PoisonMixin):
    """A general poisoning attack against logistic regression."""

    def __init__(self, boundary=None, step_size=5, max_steps=5000, opt_method='GD',
                 penalty='l2', reg_inv=1.0, solver='saga', tol=1e-4):
        """A general poisoning attack against logistic regression.

        Parameters
        ----------
        boundary
        step_size
        max_steps
        opt_method
        penalty
        reg_inv
        solver
        tol
        """

        # Boundary Properties
        self.boundary = boundary
        self.projector = Projector(boundary=boundary)

        # Logistic Regression Properties
        self.penalty = penalty.lower()
        self.solver = solver.lower()
        self.reg_inv = reg_inv

        # Attack Optimization Properties
        OptimizableAttackMixin.__init__(self, opt_method=opt_method, stepsize=step_size, n_steps=max_steps, atol=tol)

        # Data
        self.data = None
        self.labels = None
        self.attack_points = None

    @property
    def n_data(self):
        return self.data.shape[0]

    @property
    def n_features(self):
        return self.data.shape[1]

    # ############## #
    # ATTACK METHODS #
    # ############## #
    def fit(self, data, labels):
        """	Fit attack according to data and labels.

        Parameters
        ----------
        data : ndarray of shape (samples, features)
            Input data
        labels : ndarray of shape (samples,)
            Input labels
        """

        self.data = data
        self.labels = labels

        if self.boundary is None:
            self.projector.fit(data)

    def partial_fit(self, data, labels):
        """ Incremental fit on a batch of samples.

        Parameters
        ----------
        data : ndarray of shape (samples, features)
            Input data
        labels : ndarray of shape (samples,)
            Input labels
        """
        if self.data is None:
            self.data = data
            self.labels = labels
        else:
            self.data = np.append(self.data, data, axis=0)
            self.labels = np.append(self.labels, labels)

        if self.boundary is None:
            self.projector.fit(data)

    def attack(self, n_points, target_class=None):
        """ Generate a set of attack points.

        Parameters
        ----------
        n_points : int
            Number of attack points
        target_class : int
            Target class

        Returns
        -------
        opt_attack_data : ndarray
            Optimal attack points
        attack_labels : ndarray
            Attack point labels
        """

        # If we are not targeting a specific class, attack class 1
        # TODO Change this!
        if target_class is None:
            target_class = 1

        # Initialize new attack points by randomly selecting from the set of
        # training points that do not match the target label
        idx = np.nonzero(np.not_equal(self.labels, target_class))[0]
        initial_attack_data = self.data[np.random.choice(idx, size=n_points)]
        attack_labels = target_class * np.ones((n_points,))

        opt_attack_data = self.attack_optimize(initial_attack_data, attack_labels)
        #opt_attack_data = self.projector.project(opt_attack_data)
        return opt_attack_data, attack_labels

    def attack_direction(self, attack_data, attack_label, extra_data=None, extra_labels=None):
        """ Calculate Attack Direction

        For more information, see Wongrassamee 2017.

        Parameters
        ----------
        attack_data : ndarray
            Initial attack point(s)
        attack_label : ndarray
            Attack point label
        extra_data : ndarray
            Extra data to add during calculation
        extra_labels : ndarray
            Extra data to add during calculation

        Returns
        -------
        direction : ndarray
            Gradient of loss at initial attack point
        """

        # Stop optimization if it goes outside the bounds
        if self.projector.is_out_of_bounds(attack_data):
            return np.zeros_like(attack_data)

        labels = deepcopy(self.labels)

        # The attack assumes that the labels either 0 or 1, whereas we
        # internally store the labels as -1 or 1.
        labels[labels == -1] = 0
        if extra_labels is not None:
            extra_labels[extra_labels == -1] = 0
        if attack_label == -1:
            attack_label = 0

        # Fit logistic regression on union of data and attack points
        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((labels, attack_label))

        assert np.all(np.equal(np.unique(labels_union), np.array([0, 1]))), 'Assert'

        coef, intercept = self._fit_logistic_regression(data_union, labels_union)

        # Solve linear system for gradients of coef and intercept
        d_coef, d_intercept = self._model_gradients(coef, intercept,
                                                    data_union,
                                                    attack_data, attack_label)

        # Calculate gradient direction
        term_one = self._run_logistic_regression(self.data.T, coef, intercept) - labels
        term_two = self.data @ d_coef + d_intercept
        reg_term = 0.5 * (coef @ d_coef)
        direction = self.reg_inv * np.sum(np.multiply(term_one.T, term_two), axis=0) + reg_term

        # Verify the accuracy of the vectorized direction calculation
        if __debug__:
            direction_val = 0
            for ind in range(self.n_data):
                data_ind = self.data[ind].reshape(-1, 1)
                label_ind = labels[ind]
                term_one_val = self._run_logistic_regression(data_ind, coef, intercept) - label_ind
                term_two_val = data_ind.T @ d_coef + d_intercept
                direction_val += term_one_val * term_two_val
            direction_val *= self.reg_inv
            direction_val += reg_term
            assert np.allclose(direction, direction_val), 'Vectorized gradient computation incorrect!'

        return direction

    def attack_loss(self, attack_data, attack_label, extra_data=None, extra_labels=None):
        """Loss of the attacker objective at a point.

        Parameters
        ----------
        attack_data : ndarray
            Initial attack point(s)
        attack_label : ndarray
            Attack point label
        extra_data : ndarray
            Extra data to add during calculation
        extra_labels : ndarray
            Extra data to add during calculation

        Returns
        -------
        loss : float
            Loss of attacker objective
        """
        # Stop optimization if it goes outside the bounds
        if self.projector.is_out_of_bounds(attack_data):
            return - np.inf

        # Fit logistic regression on union of data and attack points
        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((self.labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((self.labels, attack_label))

        assert np.all(np.equal(np.unique(labels_union), np.array([-1, 1]))), 'Assert'

        # Fit logistic regression model on training data poisoned with attack
        # points
        coef, intercept = self._fit_logistic_regression(data_union, labels_union)

        y_true = deepcopy(self.labels)

        # Calculate the loss
        pred = (coef @ self.data.T) + intercept
        loss = (0.5 * (coef @ coef.T) + self.reg_inv * np.sum(np.log(1 + np.exp(-y_true * pred))))
        return loss

    def surrogate_model(self):
        """Get the model that is being attacked

        Returns
        -------
        model : sklearn model
            Model to being attacked
        """
        model = LogisticRegression(penalty=self.penalty,
                                   dual=False,
                                   C=self.reg_inv,
                                   fit_intercept=True,
                                   solver=self.solver,
                                   max_iter=10000,
                                   multi_class='ovr')
        return model

    ######################
    # Internal Utilities #
    ######################

    def _model_gradients(self, coef, intercept, data_union, x_c, y_c):
        """ Solve the linear system for the model gradients

        Parameters
        ----------
        coef : ndarray
            Coefficients of the trained model
        intercept : ndarray
            Intercept of the trained model
        data_union : ndarray
            Union of training data and attack points
        x_c : ndarray
            Attack point
        y_c : ndarray
            Attack labels

        Returns
        -------
        d_coef : ndarray
            Gradient of model coefficients
        d_intercept : ndarray
            Gradient of model intercept
        """

        # This method assumes the labels are 0 or 1
        assert y_c == 0 or y_c == 1, 'Assert'

        n_union = data_union.shape[0]

        # Calculate sigma, mean, and alpha that will be used in solving for
        # the model gradients
        # TODO: Implement vectorized calculations
        sigma = 0
        mean = 0
        alpha = 0
        for ind in range(n_union):
            data_ind = data_union[ind].reshape(-1, 1)
            loss_ind = self._run_logistic_regression(data_ind, coef, intercept)
            d_loss_ind = loss_ind * (1 - loss_ind)
            sigma += (d_loss_ind * (data_ind @ data_ind.T))
            mean += (d_loss_ind * data_ind)
            alpha += d_loss_ind
        sigma *= self.reg_inv
        mean *= self.reg_inv
        alpha *= self.reg_inv

        # Regularization changes the sigma term
        if self.penalty == 'l2':
            sigma += 0.5 * np.eye(self.n_features)
        else:
            raise NotImplementedError('Invalid regularization method!')

        # Calculate the loss due to the attack point
        x_c = x_c.reshape(-1, 1)
        loss_c = self._run_logistic_regression(x_c, coef, intercept)
        d_loss_c = loss_c * (1 - loss_c)

        # Calculate the m and p terms used in the linear system
        system_m = self.reg_inv * ((loss_c - y_c) * np.eye(self.n_features)) + (d_loss_c * (x_c @ coef))
        system_p = self.reg_inv * d_loss_c * coef

        # Solve the linear system
        system = np.vstack([np.hstack([sigma, mean]),
                           np.hstack([mean.T, alpha])])
        system_b = -1 * np.vstack((system_m, system_p))
        d_coef_d_intercept = np.linalg.solve(system, system_b)

        # Return the model gradients
        d_coef = d_coef_d_intercept[:self.n_features, :]
        d_intercept = d_coef_d_intercept[self.n_features:, :]
        return d_coef, d_intercept

    # ################### #
    # LOGISTIC REGRESSION #
    # ################### #
    def _fit_logistic_regression(self, data, labels):
        """Fit logistic regression model.

        Parameters
        ----------
        data : ndarray
            Input data
        labels : ndarray
            Input labels

        Returns
        -------
        coef : ndarray
            Coefficients of trained model
        intercept : ndarray
            Intercept of trained model
        """

        # Scikit-learn assumes treats the labels as -1 and 1
        labels[labels == 0] = -1
        assert np.all(np.equal(np.unique(labels), np.array([-1, 1]))), 'Assert'

        model = self.surrogate_model()
        model.fit(data, labels)
        return model.coef_, model.intercept_

    def _run_logistic_regression(self, data, coef, intercept):
        """Run standard logistic regression.

        Parameters
        ----------
        data : ndarray
            Input data
        coef : ndarray
            Fit model coefficients
        intercept : ndarray
            Fit model intercepts

        Returns
        -------
        output : ndarray
            Logistic response. Range (0, 1)
        """
        f = (coef @ data) + intercept
        return 1 / (1 + np.exp(-f))

    def _predict_logistic_regression(self, data, coef, intercept):
        """Classify data using logistic regression.

        Parameters
        ----------
        data : ndarray
            Input data
        coef : ndarray
            Fit model coefficients
        intercept : ndarray
            Fit model intercepts

        Returns
        -------
        outputs : ndarray
            Predicted labels
        """
        logits = self._run_logistic_regression(data.T, coef, intercept)

        # Binary case
        logits[logits < 0.5] = -1
        logits[logits != -1] = 1
        return logits
