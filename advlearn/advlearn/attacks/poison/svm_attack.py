"""General SVM Poison Attack"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss
from advlearn.utils import Projector
from advlearn.base import BaseAttack, OptimizableAttackMixin, PoisonMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


class SVMAttack(BaseAttack, OptimizableAttackMixin, PoisonMixin):
    """A general poisoning attack against SVM."""

    def __init__(self, boundary=None, opt_method='GD', step_size=5, max_steps=5000,
                 atol=1e-4, c=1, kernel='rbf', degree=3, coef0=1, gamma='auto'):
        """ Poisoning Support Vector Machines

        Parameters
        ----------
        boundary
        opt_method
        step_size
        max_steps
        atol
        c
        kernel
        degree
        coef0
        gamma
        """

        # Boundary Properties
        self.boundary = boundary
        self.projector = Projector(boundary=boundary)

        # SVM Properties
        self.c = c
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

        # Attack Optimization Properties
        OptimizableAttackMixin.__init__(self, opt_method=opt_method, stepsize=step_size, n_steps=max_steps, atol=atol)

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
        #opt_attack_data = self.projector.project(opt_attack_data)
        return opt_attack_data, attack_labels

    def attack_direction(self, attack_data, attack_label, extra_data=None, extra_labels=None):
        """ Calculate Attack Direction

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

        data_training = data_validation = data_union
        labels_training = labels_validation = labels_union

        # Train a support vector machine
        model = self._fit_svm(data_training, labels_training)

        # NOTE: https://www.quora.com/I-am-getting-negative-alpha-value-for-non-linear-kernel-SVM-but-the-alpha-value-should-lie-between-0-and-C-What-does-negative-alpha-value-mean
        alpha = np.zeros_like(labels_training, dtype='float64')
        alpha[model.support_] = np.abs(model.dual_coef_.flatten())

        margin_sv_ind = np.argwhere(np.logical_and(alpha > 1e-6, alpha < self.c - 1e-6))
        margin_sv_ind = margin_sv_ind.flatten()

        alpha_c = alpha[-1]

        # Evaluate label annotated kernel matrix (for both training and validation)
        if self.kernel == 'linear':
            kernal_training = self._linear_kernel(data_training, data_training)
            kernal_validation = self._linear_kernel(data_validation, data_training)
        elif self.kernel == 'poly':
            kernal_training = self._polynomial_kernel(data_training, data_training)
            kernal_validation = self._polynomial_kernel(data_validation, data_training)
        elif self.kernel == 'rbf':
            kernal_training = self._rbf_kernel(data_training, data_training)
            kernal_validation = self._rbf_kernel(data_validation, data_training)
        else:
            raise ValueError('Invalid kernel!')

        annotation_training = np.dot(np.reshape(labels_training, (-1, 1)), np.reshape(labels_training, (1, -1)))
        kernal_training = np.multiply(kernal_training, annotation_training)
        annotation_validation = np.dot(np.reshape(labels_validation, (-1, 1)), np.reshape(labels_training, (1, -1)))
        kernal_validation = np.multiply(kernal_validation, annotation_validation)

        # Determine "score" or distance to the seperating hyperplane
        # Is this score the same? https://stackoverflow.com/questions/44579826/does-the-decision-function-in-scikit-learn-return-the-true-distance-to-the-hyper
        score_raw = model.decision_function(data_validation)
        score = np.multiply(score_raw, labels_validation) - 1

        # Determine direction and take step and project onto feasible set
        direction = self._model_direction(attack_data, attack_label,
                                          data_union, labels_union,
                                          alpha_c, margin_sv_ind, score,
                                          kernal_training, kernal_validation)
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

        if extra_data is not None and extra_labels is not None:
            data_union = np.vstack((self.data, extra_data, attack_data))
            labels_union = np.hstack((self.labels, extra_labels, attack_label))
        else:
            data_union = np.vstack((self.data, attack_data))
            labels_union = np.hstack((self.labels, attack_label))

        model = self._fit_svm(data_union, labels_union)

        pred = model.decision_function(self.data)
        loss = hinge_loss(self.labels, pred)
        return loss

    def surrogate_model(self):
        """ Get the model that is being attacked

        Returns
        -------
        model : sklearn model
            Model to being attacked
        """
        model = SVC(C=self.c,
                    kernel=self.kernel,
                    degree=self.degree,
                    coef0=self.coef0,
                    gamma=self.gamma)
        return model

    ######################
    # Internal Utilities #
    ######################

    def _model_direction(self, attack_data, attack_label, data_union, labels_union, alpha_c, margin_sv_ind, score, kernal_training, kernel_validation):
        """ Calculate the gradient
        
        Parameters
        ----------
        attack_data : ndarray
            Attack data
        attack_label : ndarray
            Attack data label
        data_union : ndarray
            Union of attack points and data
        labels_union : ndarray
            Union of attack points labels and labels
        alpha_c : ndarray
            SVM dual variables
        margin_sv_ind : ndarray
            Indices of margin support vectors
        score : ndarray
            Score for each datapoint
        kernal_training : ndarray
            Training kernel matrix
        kernel_validation : ndarray
            Validation kernel matrix

        Returns
        -------
        direction : ndarray
            Direction to step
        """
        data_training = data_validation = data_union
        labels_training = labels_validation = labels_union

        # Are there no support vectors on the margin? Don't move the attack point (return zeros)
        if margin_sv_ind.size == 0:
            return np.zeros_like(attack_data)

        # Add small values to diagonal of kernel similarity matrix to improve numerical stability
        kernal_training += 1e-6 * np.eye(kernal_training.shape[0])

        # Compute R (left side of equation 7)
        y_margin = labels_training[margin_sv_ind]
        kernel_margin = kernal_training[margin_sv_ind, :][:, margin_sv_ind]
        r = np.vstack((np.hstack([np.array([0]), y_margin]),
                       np.hstack([np.reshape(y_margin, (-1, 1)), kernel_margin])))
        r = np.linalg.inv(r)

        # Kernel Grad
        if self.kernel == 'linear':
            dq_training = data_training[margin_sv_ind, :]
            dq_validation = data_validation
        elif self.kernel == 'poly':
            dq_training = self._d_polynomial_kernel(data_training[margin_sv_ind, :], attack_data)
            dq_validation = self._d_polynomial_kernel(data_validation, attack_data)
        elif self.kernel == 'rbf':
            dq_training = self._d_rbf_kernel(data_training[margin_sv_ind, :], attack_data)
            dq_validation = self._d_rbf_kernel(data_validation, attack_data)
        else:
            raise ValueError('Invalid kernel')

        # Compute y_i * x_i and y_s * x_s (for later calculations)
        x_i = np.multiply(np.reshape(labels_validation, (-1, 1)), dq_validation)
        x_s = np.multiply(np.reshape(labels_training[margin_sv_ind], (-1, 1)), dq_training)

        k = np.vstack([np.zeros((1, x_s.shape[1])), x_s])
        delta = - np.dot(r, k)

        db = delta[0, :] * alpha_c
        da = delta[1:, :] * alpha_c

        kernel_is = kernel_validation[:, margin_sv_ind]

        delta_gi = np.dot(kernel_is, da) + \
                   alpha_c * attack_label * x_i + \
                   np.dot(np.reshape(labels_validation, (-1, 1)), np.reshape(db, (1, -1)))

        delta_gi[score >= 0, :] = 0

        gradient = - np.sum(delta_gi, axis=0)
        direction = gradient / np.linalg.norm(gradient)

        if np.isnan(direction).any():
            return np.zeros_like(attack_data)

        return direction

    # ####################### #
    # SUPPORT VECTOR MACHINES #
    # ####################### #
    def _fit_svm(self, data, labels):
        """Fit support vector machine model.

        Parameters
        ----------
        data : input data. Should be an ndarray where rows are samples and
            columns are features.
        labels : input labels. Should be a 1-dimensional ndarray of
            categorical values.

        Returns
        -------
        coef : weight coefficients of trained model. Shape (features,)
        intercept : bias of trained model. Shape (1,)

        """

        # IMPORTANT!!! Scikit-learn uses -1 and 1
        # Although it will convert it internally, we will explicitly do it to make sure it does what we want!

        assert np.all(np.equal(np.unique(labels), np.array([-1, 1]))), 'Assert'

        model = self.surrogate_model()
        model.fit(data, labels)
        return model

    # ################ #
    # KERNEL FUNCTIONS #
    # ################ #
    def _linear_kernel(self, data_validation, data_training):
        """ Linear kernel function

        Parameters
        ----------
        data_validation : ndarray
            Validation data
        data_training : ndarray
            Training data

        Returns
        -------
        kernel : ndarray
            Kernel similarity matrix
        """
        return linear_kernel(data_validation, data_training)

    def _d_linear_kernel(self, data, x_c):
        """ Gradient of linear kernel with respect to attack data

        Parameters
        ----------
        data : ndarray
            Training data
        x_c : ndarray
            Attack data

        Returns
        -------
        grad : ndarray
            Gradient of kernel
        """
        return data

    def _polynomial_kernel(self, data_validation, data_training):
        """ Polynomial kernel function

        Parameters
        ----------
        data_validation : ndarray
            Validation data
        data_training : ndarray
            Training data

        Returns
        -------
        kernel : ndarray
            Kernel similarity matrix
        """

        return polynomial_kernel(data_validation, data_training, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

    def _d_polynomial_kernel(self, data, x_c):
        """Gradient of polynomial kernel with respect to attack data

        Parameters
        ----------
        data : ndarray
            Training data
        x_c : ndarray
            Attack data

        Returns
        -------
        grad : ndarray
            Gradient of kernel
        """
        dq = np.zeros_like(data)
        attack_reshape = x_c.flatten()

        for ind in range(data.shape[0]):
            first_term = (np.dot(data[ind, :], attack_reshape) + self.coef0) ** (self.degree - 1)
            dq[ind, :] = self.degree * first_term * data[ind, :]

        return dq

    def _rbf_kernel(self, data_validation, data_training):
        """Radial basis function

        Parameters
        ----------
        data_validation : ndarray
            Validation data
        data_training : ndarray
            Training data

        Returns
        -------
        kernel : ndarray
            Kernel similarity matrix
        """
        if isinstance(self.gamma, str):
            gamma = 1 / data_training.shape[1]
        else:
            gamma = self.gamma

        return rbf_kernel(data_validation, data_training, gamma)

    def _d_rbf_kernel(self, data, x_c):
        """Gradient of rbf kernel with respect to attack data

        Parameters
        ----------
        data : ndarray
            Training data
        x_c : ndarray
            Attack data

        Returns
        -------
        grad : ndarray
            Gradient of kernel
        """
        dq = np.zeros_like(data)

        attack_reshape = np.reshape(x_c, (1, -1))
        kernel = self._rbf_kernel(data, attack_reshape)
        kernel = kernel.flatten()

        for ind in range(data.shape[1]):
            dq[:, ind] = self.gamma * np.multiply(kernel, data[:, ind] - attack_reshape[0, ind])

        return dq
