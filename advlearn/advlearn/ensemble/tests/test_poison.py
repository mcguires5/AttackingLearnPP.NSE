"""Test poison ensemble"""

import numpy as np
from advlearn.ensemble.poison import PoisonEnsemble, AttackPairs
from advlearn.attacks.dummy import DummyPoisonAttack
from advlearn.classifier.dummy import DummyClassifier

X = np.array([[-0.12840393, 0.66446571], [1.32319756, -0.13181616],
              [0.04296502, -0.37981873], [0.83631853, 0.18569783],
              [1.02956816, 0.36061601], [1.12202806, 0.33811558],
              [-0.53171468, -0.53735182], [1.3381556, 0.35956356],
              [-0.35946678, 0.72510189], [1.32326943, 0.28393874],
              [2.94290565, -0.13986434], [0.28294738, -1.00125525],
              [0.34218094, -0.58781961], [-0.88864036, -0.33782387],
              [-1.10146139, 0.91782682], [-0.7969716, -0.50493969],
              [0.73489726, 0.43915195], [0.2096964, -0.61814058],
              [-0.28479268, 0.70459548], [1.84864913, 0.14729596],
              [1.59068979, -0.96622933], [0.73418199, -0.02222847],
              [0.50307437, 0.498805], [0.84929742, 0.41042894],
              [0.62649535, 0.46600596], [0.79270821, -0.41386668],
              [1.16606871, -0.25641059], [1.57356906, 0.30390519],
              [1.0304995, -0.16955962], [1.67314371, 0.19231498],
              [0.98382284, 0.37184502], [0.48921682, -1.38504507],
              [-0.46226554, -0.50481004], [-0.03918551, -0.68540745],
              [0.24991051, -1.00864997], [0.80541964, -0.34465185],
              [0.1732627, -1.61323172], [0.69804044, 0.44810796],
              [-0.5506368, -0.42072426], [-0.34474418, 0.21969797]])
Y = np.array([
    1, 2, 2, 2, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 1, 2, 0])


class TestPoisonEnsemble(object):
    """Test poison ensemble"""
    def test_poison_ensemble(self):
        classifier_1 = DummyClassifier()
        attack_1 = DummyPoisonAttack()
        classifier_2 = DummyClassifier()
        attack_2 = DummyPoisonAttack()
        attack_pairs = AttackPairs()
        attack_pairs.add(classifier_1, attack_1, 1)
        attack_pairs.add(classifier_2, attack_2, 1)
        attack_pairs.fit_all(X, Y)

        ensemble = PoisonEnsemble(attack_pairs, X)
        defender = DummyClassifier()
        ensemble.poison(defender)

class TestAttackPairs(object):
    """Test attack pairs"""

    def test_setup_empty(self):
        """Test that the list of attack pairs is empty initially"""
        attack_pairs = AttackPairs()
        assert attack_pairs.attack_pairs == list()

    def test_add_attack(self):
        """Test that an attack can be added"""
        classifier = DummyClassifier()
        attack = DummyPoisonAttack()
        attack_pairs = AttackPairs()
        attack_pairs.add(classifier, attack, 1)
        assert attack_pairs.attack_pairs == [{'classifier':classifier, 'attack':attack, 'belief':1}]

    def test_get_beliefs(self):
        """Test that you can get the current beliefs as a numpy array"""
        classifier = DummyClassifier()
        attack = DummyPoisonAttack()
        attack_pairs = AttackPairs()
        assert np.array_equal(attack_pairs.get_beliefs(), np.zeros((0,)))
        attack_pairs.add(classifier, attack, 1)
        assert np.array_equal(attack_pairs.get_beliefs(), np.array([1]))
        attack_pairs.add(classifier, attack, 1)
        assert np.array_equal(attack_pairs.get_beliefs(), np.array([1, 1]))

    def test_set_beliefs(self):
        """Test that you can set the current beliefs as a numpy array"""
        classifier = DummyClassifier()
        attack = DummyPoisonAttack()
        attack_pairs = AttackPairs()

        attack_pairs.add(classifier, attack, 1)
        rand_beliefs_1 = np.random.rand(1)
        attack_pairs.set_beliefs(rand_beliefs_1)
        assert np.array_equal(attack_pairs.get_beliefs(), rand_beliefs_1)

        attack_pairs.add(classifier, attack, 1)
        rand_beliefs_2 = np.random.rand(2)
        attack_pairs.set_beliefs(rand_beliefs_2)
        assert np.array_equal(attack_pairs.get_beliefs(), rand_beliefs_2)

        attack_pairs.add(classifier, attack, 1)
        rand_beliefs_3 = np.random.rand(3)
        attack_pairs.set_beliefs(rand_beliefs_3)
        assert np.array_equal(attack_pairs.get_beliefs(), rand_beliefs_3)

    def test_normalize_belief(self):
        """Test that the beliefs can be normalized"""
        classifier = DummyClassifier()
        attack = DummyPoisonAttack()
        attack_pairs = AttackPairs()

        attack_pairs.add(classifier, attack, 1)
        attack_pairs.add(classifier, attack, 1)
        attack_pairs.add(classifier, attack, 1)

        attack_pairs.normalize_beliefs()
        assert np.array_equal(attack_pairs.get_beliefs(), np.array([1.0, 1.0, 1.0]) / 3)

        rand_beliefs_3 = np.random.rand(3)
        attack_pairs.set_beliefs(rand_beliefs_3)
        attack_pairs.normalize_beliefs()
        assert np.array_equal(attack_pairs.get_beliefs(), rand_beliefs_3 / np.sum(rand_beliefs_3))

    def test_fit_predict_all(self):
        """Test that you can fit and predict on all classifiers"""
        classifier_1 = DummyClassifier()
        attack_1 = DummyPoisonAttack()
        classifier_2 = DummyClassifier()
        attack_2 = DummyPoisonAttack()
        attack_pairs = AttackPairs()
        attack_pairs.add(classifier_1, attack_1, 1)
        attack_pairs.add(classifier_2, attack_2, 1)
        attack_pairs.fit_all(X, Y)
        y_out = attack_pairs.predict_all(X)
        assert isinstance(y_out, np.ndarray)
        assert y_out.shape == (X.shape[0], 2)

    def test_get_attack_point(self):
        """Test that you can get an attack point from a particular attack"""
        classifier_1 = DummyClassifier()
        attack_1 = DummyPoisonAttack()
        classifier_2 = DummyClassifier()
        attack_2 = DummyPoisonAttack()
        attack_pairs = AttackPairs()
        attack_pairs.add(classifier_1, attack_1, 1)
        attack_pairs.add(classifier_2, attack_2, 1)
        attack_pairs.fit_all(X, Y)

        (x_attack_1, y_attack_1) = attack_pairs.get_attack_point(0)
        assert isinstance(x_attack_1, np.ndarray)
        assert x_attack_1.shape == (1, 2)

        (x_attack_2, y_attack_2) = attack_pairs.get_attack_point(1)
        assert isinstance(x_attack_1, np.ndarray)
        assert x_attack_1.shape == (1, 2)
