"""Outlier Detection Figure"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.datasets.samples_generator import make_blobs

from advlearn.attacks.poison import GeneralTikhonovAttack


# http://scikit-learn.org/stable/modules/outlier_detection.html
# http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html

##############
# Parameters #
##############
LOAD_ATTACK_POINTS = True

distance_thresholds = np.linspace(0, 9, num=10)
num_attack = distance_thresholds.size
num_test = 40

boundary_mins = (0, -20, -20)
boundary_maxs = (20, 20, 20)
step_sizes = (10000, 1000, 1000)
num_steps = (1000, 1000, 1000)

# Distance Threshold Outlier Term
outlier_method = 'distancethreshold'
outlier_weight = 1
outlier_distance_threshold = 1
outlier_k = 1

# gaussian dataset
centers = [[-1, 0], [1, 0]]
Xtr, ytr = make_blobs(n_samples=50, centers=centers, cluster_std=0.5, n_features=2, random_state=0)
Xte, yte = make_blobs(n_samples=50, centers=centers, cluster_std=0.5, n_features=2, random_state=1)
datasets = {
    'spambase': {
        'train': np.loadtxt('data/datasets/spambase_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/spambase_test.csv', delimiter=',')
    },
    'congressional_voting': {
        'train': np.loadtxt('data/datasets/congressional-voting_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/congressional-voting_test.csv', delimiter=',')
    },
    'credit_approval': {
        'train': np.loadtxt('data/datasets/credit-approval_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/credit-approval_test.csv', delimiter=',')
    }
    # 'gaussian': {
    #     'train': np.concatenate((Xtr, ytr.reshape(-1, 1)), axis=1),
    #     'test': np.concatenate((Xte, yte.reshape(-1, 1)), axis=1)
    # },
}

classifiers = {
    'Distance Threshold': None,
    'One-Class SVM': svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1/(num_test + 1)),
    'Isolation Forest': IsolationForest(contamination=num_attack/(num_test + 1)),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=5)
}

plt.figure(figsize=(12, 11))
for i, (dataset_name, dataset) in enumerate(datasets.items()):

    print('in dataset {:s}'.format(dataset_name))

    x_train = dataset['train'][:, :-1]
    y_train = (dataset['train'][:, -1] * 2) - 1
    x_test_full = dataset['test'][:, :-1]
    use = np.random.choice(x_test_full.shape[0], num_test)
    x_test = x_test_full[use, :]

    # Generate the Attack Points
    boundary = np.concatenate((boundary_mins[i] * np.ones((1, x_test.shape[1])),
                               boundary_maxs[i] * np.ones((1, x_test.shape[1]))))
    if outlier_method == 'distancethreshold':
        attack = GeneralTikhonovAttack(boundary=boundary,
                                       step_size=step_sizes[i],
                                       max_steps=num_steps[i],
                                       outlier_method='distancethreshold',
                                       outlier_weight=outlier_weight,
                                       outlier_distance_threshold=outlier_distance_threshold,
                                       outlier_k=outlier_k)
    else:
        attack = GeneralTikhonovAttack(boundary=boundary, step_size=step_sizes[i], max_steps=num_steps[i])

    print('fitting attack to training data')
    attack.fit(x_train, y_train)
    successful_load = True
    if LOAD_ATTACK_POINTS:
        print('loading attack points')
        try:
            attack_points = np.loadtxt('cache/{}_attack_points.csv'.format(dataset_name), delimiter=',')
            attack_point_labels = np.loadtxt('cache/{}_attack_point_labels.csv'.format(dataset_name), delimiter=',')
        except:
            print('failed to load attack points, generating')
            successful_load = False

    if not (LOAD_ATTACK_POINTS and successful_load):
        attack_points = np.zeros((num_attack, x_train.shape[1]))
        attack_point_labels = np.zeros(num_attack)
        for a in range(num_attack):
            print('getting attack point {:d} of {:d}'.format(a+1, num_attack))

            attack.outlier_distance_threshold = distance_thresholds[a]
            print('Dt: {}\n'.format(distance_thresholds[a]))
            x_attack, y_attack = attack.get_attack_point()
            attack_points[a, :] = x_attack
            attack_point_labels[a] = y_attack

        np.savetxt('cache/{}_attack_points.csv'.format(dataset_name), attack_points, delimiter=',')
        np.savetxt('cache/{}_attack_point_labels.csv'.format(dataset_name), attack_point_labels, delimiter=',')

    print(attack_points)

    x_test_labels = np.ones(x_test[:, -1].shape)
    x_test_poison = np.concatenate((x_test, attack_points))
    x_test_poison_labels = np.concatenate((x_test_labels, -1 * np.ones(attack_points[:, -1].shape)))

    # get other metric
    other_metric = np.zeros(num_attack)
    for k in range(num_attack):
        attack2 = GeneralTikhonovAttack(boundary=boundary, step_size=step_sizes[i], max_steps=num_steps[i])
        attack2.fit(x_train, y_train)
        other_metric[k] = attack2.attacker_objective(attack_points[k, :], attack_point_labels[k])
        # lasso = Lasso(alpha=0.1)
        # x_train_poison = np.concatenate((x_train, [attack_points[k, :]]))
        # x_train_poison_labels = np.concatenate((y_train, [attack_point_labels[k]]))
        # lasso.fit(x_train_poison, x_train_poison_labels)
        # x_pred_labels = lasso.predict(x_test)
        # other_metric[k] = np.mean(np.equal(x_test_labels, np.sign(x_pred_labels)))
        # other_metric[k] = np.sqrt(mean_squared_error(y_true=x_test_labels, y_pred=x_pred_labels))
    print(other_metric)

    for j, (classifier_name, classifier) in enumerate(classifiers.items()):
        print('Using Dataset {} and Classifier {}'.format(dataset_name, classifier_name))

        # Fit Classifier on Data and Calculate Outlier Score
        # Normalize these so that positive is more abnormal
        if classifier_name == 'Distance Threshold':
            nbrs = NearestNeighbors(n_neighbors=1).fit(x_train)
            dists, _ = nbrs.kneighbors(x_test_poison)
            outlier_score = dists
        elif classifier_name == 'Local Outlier Factor':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier._decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        elif classifier_name == 'Isolation Forest':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier.decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        elif classifier_name == 'One-Class SVM':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier.decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        else:
            raise ValueError('No valid classifier name!')

        # get outlier scores
        outlier_score = outlier_score.reshape((-1,))
        outlier_score_test = outlier_score[:-num_attack-1]
        outlier_score_attack = outlier_score[-num_attack:]

        outlier_score_test = np.sort(outlier_score_test)

        # Save Data
        np.savetxt('data/outlier_vs_attack_strength/f3_{}_{}.csv'.format(classifier_name, dataset_name),
                   np.vstack((distance_thresholds, outlier_score_attack)).T)

        # Plot Data
        ax1 = plt.subplot(len(datasets), len(classifiers), i * (len(datasets) + 1) + j + 1)
        # plt.vlines(range(outlier_score_test.size), ymin=0, ymax=outlier_score_test)
        # plt.scatter(range(outlier_score_test.size), outlier_score_test)
        ax1.plot(distance_thresholds, outlier_score_attack)

        ax2 = ax1.twinx()
        ax2.plot(distance_thresholds, other_metric, 'r')

        if i == 0:
            plt.title(classifier_name)
        if j == 0:
            plt.ylabel(dataset_name)

plt.show()

print('done')
