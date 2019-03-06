
import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from advlearn.attacks.poison import GeneralTikhonovAttack
from advlearn.pipeline import Pipeline
from advlearn.ensemble.poison import PoisonEnsemble, AttackPairs
from advlearn.defenses import KthNeighbor
from advlearn.classifier import OnlineLasso

num_steps = 10
alpha_lambda = 0.1
distance_threshold = 1

datasets = {
    'spambase': {
        'train': np.loadtxt('csv/spambase_train.csv', delimiter=','),
        'test': np.loadtxt('csv/spambase_train.csv', delimiter=',')
    },
    'credit-approval': {
        'train': np.loadtxt('csv/credit-approval_train.csv', delimiter=','),
        'test': np.loadtxt('csv/credit-approval_test.csv', delimiter=',')
    },
    'congressional-voting': {
        'train': np.loadtxt('csv/congressional-voting_train.csv', delimiter=','),
        'test': np.loadtxt('csv/congressional-voting_test.csv', delimiter=',')
    }
}

dataset_name = 'spambase'
data_train_x = datasets[dataset_name]['train'][:, :-2]
data_train_y = (datasets[dataset_name]['train'][:, -1] * 2) - 1
data_test_x = datasets[dataset_name]['test'][:, :-2]
data_test_y = (datasets[dataset_name]['test'][:, -1] * 2) - 1

# Setup Defender
defender = OnlineLasso(alpha=alpha_lambda)

# Setup Classifiers
lasso = OnlineLasso(alpha=alpha_lambda)
steps = [('KthNeighbor', KthNeighbor(outlier_distance_threshold=distance_threshold)),
         ('Lasso', OnlineLasso(alpha=alpha_lambda))]
lasso_outlier = Pipeline(steps)

# Setup Attacks
attack_lasso = GeneralTikhonovAttack(lasso_lambda=alpha_lambda,
                                     step_size=5000,
                                     max_steps=1000)
attack_lasso_outlier = GeneralTikhonovAttack(lasso_lambda=alpha_lambda,
                                             step_size=5000,
                                             max_steps=1000,
                                             outlier_method='distancethreshold',
                                             outlier_distance_threshold=distance_threshold)

# Setup Attack Pairs
attack_pairs = AttackPairs()
attack_pairs.add(lasso, attack_lasso, 1)
attack_pairs.add(lasso_outlier, attack_lasso_outlier, 1)
attack_pairs.fit_all(data_train_x, data_train_y)

# Setup Ensemble
ensemble = PoisonEnsemble(attack_pairs, data_test_x, defender=defender)

beliefs = np.zeros((num_steps+1, 2))
beliefs[0, :] = np.array([0.5, 0.5])
for t in range(num_steps):
    ensemble.poison_single()
    beliefs[t+1, :] = ensemble.attack_pairs.get_beliefs()

print(beliefs)

plt.figure(figsize=(9, 9))
plt.stackplot(range(num_steps+1), beliefs[:, 0], beliefs[:, 1], labels=["Lasso", "Lasso Outlier"], baseline='zero')
plt.legend(loc=2)
plt.title('Ensemble Beliefs')
plt.show()

