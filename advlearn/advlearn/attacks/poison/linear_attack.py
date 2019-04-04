"""General Linear Poison Attack"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from advlearn.base import (
    BaseAttack,
    OptimizableAttackMixin,
    IntelligentAttackMixin,
    PoisonMixin,
)
from advlearn.utils import Projector


class LinearAttack(
    BaseAttack, OptimizableAttackMixin, IntelligentAttackMixin, PoisonMixin
):
    """A general poisoning attack against linear regression.

    Defender regularization may be none (linear), L1 (LASSO), L2 (ridge),
    or elastic net.

    Outlier detection may be distance threshold or kth-nearest neighber."""

    def __init__(
        self,
        boundary=None,
        lasso_lambda=0.1,
        ridge_alpha=1,
        elasticnet_alpha=1,
        elasticnet_l1_ratio=0.5,
        efs_method="lasso",
        opt_method="GD",
        step_size=5,
        max_steps=5000,
        atol=1e-4,
        outlier_method=None,
        outlier_weight=1,
        outlier_distance_threshold=1,
        outlier_power=2,
        outlier_k=3,
        surface_size=40,
    ):
        """General Linear Poison Attack

        Parameters
        ----------
        lasso_lambda : lasso regression free parameter
        ridge_alpha : ridge regression free parameter
        elasticnet_alpha : elastic net free parameter
        elasticnet_l1_ratio : elastic net free parameter
        efs_method : regularization / embedded feature selection method
        step_size : optimization step size
        max_steps : optimization maximum number of steps
        outlier_method : outlier term method
        outlier_weight : outlier term weight
        outlier_distance_threshold : outlier term distance threshold
        outlier_power : outlier term power
        outlier_k : outlier term kth nearest neighbor
        surface_size : size of visualization surface
        """

        # Boundary Properties
        self.boundary = boundary
        self.projector = Projector(boundary=boundary)

        # Linear Regression Properties
        self.lasso_lambda = lasso_lambda
        self.ridge_alpha = ridge_alpha
        self.elasticnet_alpha = elasticnet_alpha
        self.elasticnet_l1_ratio = elasticnet_l1_ratio
        self.efs_method = efs_method

        # Attack Optimization Properties
        OptimizableAttackMixin.__init__(
            self,
            opt_method=opt_method,
            stepsize=step_size,
            n_steps=max_steps,
            atol=atol,
        )

        # Intelligent Attack Properties
        IntelligentAttackMixin.__init__(
            self,
            outlier_method=outlier_method,
            outlier_weight=outlier_weight,
            outlier_distance_threshold=outlier_distance_threshold,
            outlier_power=outlier_power,
            outlier_k=outlier_k,
        )

        # Other init
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
        return

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
        # opt_attack_data = self.projector.project(opt_attack_data)
        return opt_attack_data, attack_labels

    def attack_direction(
        self, attack_data, attack_label, extra_data=None, extra_labels=None
    ):
        """Calculate the gradient.

        Reference: Xiao et al. 2015.

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

        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((self.labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((self.labels, attack_label))

        coef, intercept = self._fit_linear_regression(data_union, labels_union)

        # Compute grad(W)
        dcoef, dintercept = self._model_gradients(
            coef, intercept, attack_data, attack_label
        )
        if self.efs_method is None:
            reg_term = 0
        elif self.efs_method == "lasso":
            reg_term = self.lasso_lambda * (np.sign(coef) @ dcoef)
        elif self.efs_method == "ridge":
            reg_term = self.lasso_lambda * (coef @ dcoef)
        elif self.efs_method == "elasticnet":
            reg_term = self.lasso_lambda * (
                (
                    self.elasticnet_l1_ratio * np.sign(coef)
                    + (1 - self.elasticnet_l1_ratio) * coef
                )
                @ dcoef
            )
        else:
            raise ValueError("Invalid embedded feature selection method.")

        grad_w_diff = np.tile(
            self._run_linear_regression(self.data, coef, intercept) - self.labels,
            (self.n_features, 1),
        )
        grad_w_sum = self.data @ dcoef + np.tile(dintercept, (self.n_data, 1))

        direction = np.mean(np.multiply(grad_w_sum, grad_w_diff.T), axis=0) + reg_term
        return direction

    def attack_loss(
        self, attack_data, attack_label, extra_data=None, extra_labels=None
    ):
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
            return -np.inf

        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((self.labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((self.labels, attack_label))

        coef, intercept = self._fit_linear_regression(data_union, labels_union)
        pred = self._run_linear_regression(self.data, coef, intercept)
        loss = mean_squared_error(self.labels, pred)

        return loss

    def surrogate_model(self):
        """Get the model that is being attacked

        Returns
        -------
        model : sklearn model
            Model to being attacked
        """
        if self.efs_method is None:
            model = LinearRegression(fit_intercept=True)
        elif self.efs_method == "lasso":
            model = Lasso(alpha=self.lasso_lambda, fit_intercept=True)
        elif self.efs_method == "ridge":
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        elif self.efs_method == "elasticnet":
            model = ElasticNet(
                alpha=self.elasticnet_alpha,
                l1_ratio=self.elasticnet_l1_ratio,
                fit_intercept=True,
            )
        else:
            raise ValueError("Invalid efs_method parameter!")

        return model

    ######################
    # Internal Utilities #
    ######################
    def _model_gradients(self, coef, intercept, x_c, y_c):
        """Solve the linear system for the model gradients

        Parameters
        ----------
        coef : ndarray
            Model coefficients
        intercept : ndarray
            Model intercept
        x_c : ndarray
            Attack point
        y_c : ndarray
            Attack point label

        Returns
        -------
        d_coef : ndarray
            Gradient of model coefficients
        d_intercept : ndarray
            Gradient of model intercept
        """
        """Reference: Xiao et al. 2015."""
        if self.efs_method is None or self.efs_method == "lasso":
            sigma = (1.0 / self.n_data) * self.data.T @ self.data
        elif self.efs_method == "ridge":
            sigma = (
                (1.0 / self.n_data) * self.data.T @ self.data
            ) + self.ridge_alpha * np.eye(self.n_features)
        elif self.efs_method == "elasticnet":
            sigma = ((1.0 / self.n_data) * self.data.T @ self.data) + (
                self.elasticnet_alpha
                * (1 - self.elasticnet_l1_ratio)
                * np.eye(self.n_features)
            )
        else:
            raise ValueError("Invalid embedded feature selection method.")

        mean = np.mean(self.data, axis=0).reshape((-1, 1))
        kkt_m = x_c.reshape(-1, 1) @ coef.reshape(1, -1) + np.eye(self.n_features) * (
            (x_c @ coef + intercept) - y_c
        )

        kkt_a = np.vstack(
            [np.hstack([sigma, mean]), np.hstack([mean.T, np.array([[1]])])]
        )
        kkt_b = -(1.0 / self.n_data) * np.vstack((kkt_m, coef.T))

        dcoef_dintercept = np.linalg.solve(kkt_a, kkt_b)

        d_coef = dcoef_dintercept[:-1, :]
        d_intercept = dcoef_dintercept[-1, :]

        return d_coef, d_intercept

    #####################
    # LINEAR REGRESSION #
    #####################
    def _run_linear_regression(self, data, coef, intercept):
        """Run standard linear regression

        Parameters
        ----------
        data : ndarray
            Input data
        coef : ndarray
            Regression coefficients
        intercept : error term ()

        Returns
        -------
        y_res : ndarray
            Output response
        """
        return np.dot(data, coef.reshape((-1,))) + intercept

    def _predict_linear_regression(self, data, coef, intercept):
        """Classify data using linear regression

        Parameters
        ----------
        data : ndarray
            Input data
        coef : ndarray
            Regression coefficients
        intercept : ndarray
            Error term

        Returns
        -------
        y : ndarray
            Output labels
        """
        return np.sign(self._run_linear_regression(data, coef, intercept))

    def _fit_linear_regression(self, data, labels):
        """Fit linear regression model

        Parameters
        ----------
        data : ndarray
            Input data
        labels : ndarray
            Input labels (must to -1 or 1)

        Returns
        -------
        coef : ndarray
            Regression coefficients
        intercept : ndarray
            Error term

        """
        model = self.surrogate_model()
        model.fit(data, labels)
        coef = model.coef_
        intercept = model.intercept_

        return coef, intercept
