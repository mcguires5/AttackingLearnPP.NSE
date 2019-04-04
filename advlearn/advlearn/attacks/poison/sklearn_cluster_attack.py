"""General Logistic Poison Attack"""

import numpy as np
from sklearn.base import clone
from advlearn.base import BaseAttack, PoisonMixin, OptimizableAttackMixin
from advlearn.utils import Projector
from sklearn.metrics import log_loss
from sklearn.metrics.cluster import adjusted_rand_score


class SklearnClusterAttack(BaseAttack, OptimizableAttackMixin, PoisonMixin):
    """A general poisoning attack against logistic regression."""

    def __init__(
        self,
        model,
        boundary=None,
        step_size=5,
        max_steps=5000,
        opt_method="GD",
        atol=1e-4,
    ):
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

        self.model = model

        # Boundary Properties
        self.boundary = boundary
        self.projector = Projector(boundary=boundary)

        # Decision Tree Properties

        # Attack Optimization Properties
        OptimizableAttackMixin.__init__(
            self,
            opt_method=opt_method,
            stepsize=step_size,
            n_steps=max_steps,
            atol=atol,
        )

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
        # opt_attack_data = self.projector.project(opt_attack_data)
        return opt_attack_data, attack_labels

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

        # Fit logistic regression on union of data and attack points
        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((self.labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((self.labels, attack_label))

        # Fit logistic regression model on training data poisoned with attack
        # points
        model = self.surrogate_model()
        model.fit(data_union, labels_union)

        # Calculate the loss
        try:
            preds = model.predict(self.data)
        except AttributeError:
            preds = model.fit_predict(self.data)
        loss = adjusted_rand_score(self.labels, preds)
        return loss

    def surrogate_model(self):
        """Get the model that is being attacked

        Returns
        -------
        model : sklearn model
            Model to being attacked
        """
        model = clone(self.model)
        return model
