"""Poison ensemble"""

import numpy as np


class PoisonEnsemble(object):
    """Poison ensemble"""

    def __init__(self, attack_pairs, candidate_test_points, defender=None):
        """Poison ensemble

        Parameters
        ----------
        attack_pairs : must be pretrained
        candidate_test_points
        """
        self.attack_pairs = attack_pairs
        self.candidate_test_points = candidate_test_points
        self.defender = defender

    def poison(self, defender, num_steps=10):
        """Poison defender

        Parameters
        ----------
        defender
        num_steps
        """
        self.defender = defender
        for _ in range(0, num_steps):
            self.poison_single()

    def poison_single(self):
        """Run one poison step"""

        print("New Run!")
        self.attack_pairs.normalize_beliefs()

        # Randomly select an attack
        attack_index = np.random.choice(
            np.arange(len(self.attack_pairs)), p=self.attack_pairs.get_beliefs()
        )
        print("Attack Index")
        print(attack_index)

        # Generate the attack point
        (x_attack, y_attack) = self.attack_pairs.get_attack_point(attack_index)
        print("Attack Point")
        print(x_attack)
        print(y_attack)

        # Train all classifiers on attack point
        self.attack_pairs.fit_all(x_attack, y_attack)

        # Predict all classifiers on all candidate test points
        y_test_points = self.attack_pairs.predict_all(self.candidate_test_points)

        # Calculate label variance
        var_test_points = np.var(y_test_points, axis=1)
        print("Max Variance")
        print(np.nanmax(var_test_points))

        # Determine test point with maximum label variance
        test_point_index = np.nanargmax(var_test_points)
        x_test_point = np.reshape(
            self.candidate_test_points[test_point_index, :], (1, -1)
        )
        y_test_point = y_test_points[test_point_index, :]
        print("Test Point")
        print(x_test_point)
        print("Test Point Prediction")
        print(y_test_point)

        # Send attack and test points to the defender
        self.defender.fit(x_attack, y_attack)
        y_defender = self.defender.predict(x_test_point)
        print("Defender Prediction")
        print(y_defender)

        # Update beliefs
        new_beliefs = self.attack_pairs.get_beliefs()
        new_beliefs[y_defender == y_test_point] = (
            2 * new_beliefs[y_defender == y_test_point]
        )
        self.attack_pairs.set_beliefs(new_beliefs)
        self.attack_pairs.normalize_beliefs()

        print("Beliefs")
        print(self.attack_pairs.get_beliefs())


class AttackPairs(object):
    """Attack pairs"""

    def __init__(self, attack_pairs=None):
        # TODO Add input validation
        if attack_pairs:
            self.attack_pairs = attack_pairs
        else:
            self.attack_pairs = list()

    def __len__(self):
        return len(self.attack_pairs)

    def add(self, classifier, attack, belief=1):
        """Add an attack pair

        Parameters
        ----------
        classifier : classifier object in pair
        attack : attack object in pair
        belief : current belief that the defender is using this classifier
        """
        self.attack_pairs.append(
            {"classifier": classifier, "attack": attack, "belief": belief}
        )

    def get_beliefs(self):
        """Get the current beliefs as a numpy array

        Returns
        -------
        beliefs : np.ndarray
        """
        beliefs = np.zeros((len(self.attack_pairs),))
        for index, attack_pair in enumerate(self.attack_pairs):
            beliefs[index] = attack_pair.get("belief")
        return beliefs

    def set_beliefs(self, beliefs):
        """Set the current beliefs as a numpy array

        Parameters
        ----------
        beliefs
        """
        for index, attack_pair in enumerate(self.attack_pairs):
            attack_pair["belief"] = beliefs[index]

    def normalize_beliefs(self):
        """Normalize the current beliefs so that they sum to one
        """
        beliefs = self.get_beliefs()
        self.set_beliefs(beliefs / np.sum(beliefs))

    def fit_all(self, X, y):
        """Fit all classifiers and attacks on data

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        """
        for attack_pair in self.attack_pairs:
            attack_pair["classifier"].fit(X, y)
            attack_pair["attack"].fit(X, y)

    def predict_all(self, X):
        """Predict all classifiers on data

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        y_out : np.ndarray
        """
        y_out = np.zeros((X.shape[0], len(self.attack_pairs)))
        for index, attack_pair in enumerate(self.attack_pairs):
            y_out[:, index] = attack_pair["classifier"].predict(X)
        return y_out

    def get_attack_point(self, index):
        """Get attack point of attack with index

        Parameters
        ----------
        index : the attack to generate the attack point from

        Returns
        -------
        x_attack : np.ndarray
        """
        return self.attack_pairs[index]["attack"].get_attack_point()
