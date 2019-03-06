"""Outlier Detection Figure"""

import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from advlearn.attacks.poison import GeneralTikhonovAttack


# http://scikit-learn.org/stable/modules/outlier_detection.html
# http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html

##############
# Parameters #
##############
num_attack = 2
num_test = 40

boundary_mins = (0, -20, -20)
boundary_maxs = (20, 20, 20)
step_sizes = (10000, 1000, 1000)
num_steps = (1000, 1000, 1000)

lasso_lambda = 0.1

# No Outlier Term
outlier_method = 'none'

# Distance Threshold Outlier Term
# outlier_method = 'distancethreshold'
# outlier_weight = 1
# outlier_distance_threshold = 1
# outlier_k = 1

datasets = {
    'spambase': {
        'train': np.loadtxt('data/datasets/spambase_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/spambase_train.csv', delimiter=',')
    },
    'credit_approval': {
        'train': np.loadtxt('data/datasets/credit-approval_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/credit-approval_test.csv', delimiter=',')
    },
    'congressional_voting': {
        'train': np.loadtxt('data/datasets/congressional-voting_train.csv', delimiter=','),
        'test': np.loadtxt('data/datasets/congressional-voting_test.csv', delimiter=',')
    }
}


classifiers = {
    'dt': None,
    'svm': svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1/(num_test + 1)),
    'if': IsolationForest(contamination=num_attack/(num_test + 1)),
    'lof': LocalOutlierFactor(n_neighbors=5)
}

plt.figure(figsize=(12, 11))
for i, (dataset_name, dataset) in enumerate(datasets.items()):

    x_train = dataset['train'][:, :-2]
    y_train = (dataset['train'][:, -1] * 2) - 1
    x_test_full = dataset['test'][:, :-2]
    use = np.random.choice(x_test_full.shape[0], num_test)
    x_test = x_test_full[use, :]

    # Generate the Attack Points
    boundary = np.concatenate((boundary_mins[i] * np.ones((1, x_test.shape[1])),
                               boundary_maxs[i] * np.ones((1, x_test.shape[1]))))
    if outlier_method == 'distancethreshold':
        attack = GeneralTikhonovAttack(boundary=boundary,
                                       step_size=step_sizes[i],
                                       max_steps=num_steps[i],
                                       lasso_lambda=lasso_lambda,
                                       outlier_method='distancethreshold',
                                       outlier_weight=outlier_weight,
                                       outlier_distance_threshold=outlier_distance_threshold,
                                       outlier_k=outlier_k)
    else:
        attack = GeneralTikhonovAttack(boundary=boundary, step_size=step_sizes[i],
                                       max_steps=num_steps[i], lasso_lambda=lasso_lambda)

    attack.fit(x_train, y_train)
    attack_points = np.zeros((num_attack, x_train.shape[1]))
    for a in range(num_attack):
        print('generating attack point {} of {} for {}'.format(a+1, num_attack, dataset_name))
        x_attack, y_attack = attack.get_attack_point()
        attack_points[a, :] = x_attack

    print(attack_points)

    x_test_poison = np.concatenate((x_test, attack_points))
    x_test_poison_labels = np.concatenate((np.ones(x_test[:, -1].shape), -1 * np.ones(attack_points[:, -1].shape)))

    for j, (classifier_name, classifier) in enumerate(classifiers.items()):
        print('Using Dataset {} and Classifier {}'.format(dataset_name, classifier_name))

        # Fit Classifier on Data and Calculate Outlier Score
        # Normalize these so that positive is more abnormal
        if classifier_name == 'dt':
            nbrs = NearestNeighbors(n_neighbors=1).fit(x_train)
            dists, _ = nbrs.kneighbors(x_test_poison)
            outlier_score = dists
        elif classifier_name == 'lof':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier._decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        elif classifier_name == 'if':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier.decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        elif classifier_name == 'svm':
            # We wan't the outlier score and not the decision function.
            # Take the max and subtract.
            # http://activisiongamescience.github.io/2015/12/23/Unsupervised-Anomaly-Detection-SOD-vs-One-class-SVM/
            classifier.fit(x_train)
            outlier_score = classifier.decision_function(x_test_poison)
            outlier_score = np.max(outlier_score) - outlier_score
        else:
            raise ValueError('No valid classifier name!')

        outlier_score = outlier_score.reshape((-1,))

        sort_ind = np.argsort(outlier_score)
        outlier_score = outlier_score[sort_ind]
        x_test_poison_labels = x_test_poison_labels[sort_ind]

        data_normal = np.hstack((
            np.array((np.where(x_test_poison_labels == 1))).reshape((-1, 1)),
                     outlier_score[x_test_poison_labels == 1].reshape((-1, 1))
        ))
        np.savetxt('data/dataset_outlier/f1_norm_{}_{}.csv'.format(classifier_name, dataset_name), data_normal, delimiter=',')
        data_outlier = np.hstack((
            np.array((np.where(x_test_poison_labels == -1))).reshape((-1, 1)),
            outlier_score[x_test_poison_labels == -1].reshape((-1, 1))
        ))
        np.savetxt('data/dataset_outlier/f1_out_{}_{}.csv'.format(classifier_name, dataset_name), data_outlier, delimiter=',')

        x_test_poison = x_test_poison[sort_ind, :]
        print(x_test_poison[-1, :])

        subplot = plt.subplot(len(datasets), len(classifiers), i * (len(datasets) + 1) + j + 1)

        plt_color = np.where(x_test_poison_labels == 1, 'b', 'r')
        plt.vlines(range(outlier_score.size), ymin=0, ymax=outlier_score, color=plt_color)
        plt.scatter(range(outlier_score.size), outlier_score, color=plt_color)

        if i == 0:
            plt.title(classifier_name)
        if j == 0:
            plt.ylabel(dataset_name)

plt.show()
