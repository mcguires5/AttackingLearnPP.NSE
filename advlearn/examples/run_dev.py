# Development misc tests
# Basically all the junk to run during development

from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from advlearn.ensemble.poison import AttackPairs, PoisonEnsemble
from advlearn.attacks.poison import LinearAttack
from advlearn.classifier import OnlineLasso
from advlearn.defenses import KthNeighbor
from advlearn.pipeline import Pipeline
import matplotlib
import matplotlib.pyplot as plt

# Generate Data
centers = [[-1, 0], [1, 0]]
X, y = make_blobs(n_samples=100, centers=centers, cluster_std=0.4, n_features=2, random_state=0)
y = (y*2)-1

matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], c=y)
ax.set_autoscale_on(True)
ax.set_title('Toy Gaussian Dataset')
plt.show()

# Attack Stuff
attack_pairs = AttackPairs()

# Setup Regular Lasso
lasso_classifier = OnlineLasso()
lasso_attack = LinearAttack(boundary=np.array([[-10, -10], [10, 10]]),
                            lasso_lambda=0.01)
attack_pairs.add(lasso_classifier, lasso_attack, 1)

# Setup Outlier Lasso
steps = [('KthNeighbor', KthNeighbor(outlier_distance_threshold=1)),
         ('OnlineLasso', OnlineLasso())]
lasso_outlier_classifier = Pipeline(steps)
lasso_outlier_attack = LinearAttack(boundary=np.array([[-10, -10], [10, 10]]),
                                    lasso_lambda=0.01,
                                    outlier_method='distancethreshold',
                                    outlier_distance_threshold=1)
attack_pairs.add(lasso_outlier_classifier, lasso_outlier_attack, 1)

attack_pairs.fit_all(X, y)
ensemble = PoisonEnsemble(attack_pairs, X)

defender = OnlineLasso()
defender.fit(X, y)

for _ in range(0,10):
    ensemble.poison(defender, num_steps=1)

    matplotlib.rcParams['axes.unicode_minus'] = False

    ax = plt.subplot(3, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], c=ensemble.defender.predict(X))
    ax.set_autoscale_on(True)
    ax.set_title('Defender')

    ax = plt.subplot(3, 1, 2)
    ax.scatter(X[:,0], X[:,1], c=ensemble.attack_pairs.attack_pairs[0].get('classifier').predict(X))
    ax.set_autoscale_on(True)
    ax.set_title('Regular Lasso')

    ax = plt.subplot(3, 1, 3)
    ax.scatter(X[:,0], X[:,1], c=ensemble.attack_pairs.attack_pairs[1].get('classifier').predict(X))
    ax.set_autoscale_on(True)
    ax.set_title('Outlier Lasso')
    plt.show()

# Make sure plot will not close
plt.show()