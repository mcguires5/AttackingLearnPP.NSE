"""Generate gaussian attack strength curve Figure"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

from advlearn.attacks.poison import GeneralTikhonovAttack
from advlearn.pipeline import Pipeline
from advlearn.defenses import KthNeighbor



##############
# Parameters #
##############
attacker_outlier_distance_thresholds = np.arange(0, 10.1, .25)
defender_outlier_distance_thresholds = np.array([None, 5, 3, 1])
num_runs_error = 50

alpha_lambda = 0.1

x_boundary_min = -100
x_boundary_max = 100
y_boundary_min = -100
y_boundary_max = 100

# start attack point just barely on the red side
attack_starting_position = np.array([[0.2, 0]])

# Generate Boundary
boundary = np.array([[x_boundary_min, y_boundary_min], [x_boundary_max, y_boundary_max]])


def create_toy_data(random_state=None):
    centers = [[-1, 0], [1, 0]]
    X, y = make_blobs(n_samples=50, centers=centers, cluster_std=0.5, n_features=2, random_state=random_state)
    y = (y * 2) - 1

    # make sure there is a blue point at the attack starting point so that we don't start in outlier region
    X = np.append(X, attack_starting_position, axis=0)
    y = np.append(y, np.array([-1]))
    return X, y


def fit_attack(X, y, outlier_distance_threshold=1):
    attack = GeneralTikhonovAttack(step_size=5, max_steps=5000,
                                   lasso_lambda=alpha_lambda,
                                   boundary=boundary,
                                   outlier_method='distancethreshold',
                                   outlier_distance_threshold=outlier_distance_threshold)
    attack.fit(X, y, starting_position=attack_starting_position)
    return attack


def fit_predict_defender(X, y, outlier_distance_threshold=None):
    if outlier_distance_threshold is None:
        lasso = Lasso(alpha=alpha_lambda)
    else:
        steps = [('KthNeighbor', KthNeighbor(outlier_distance_threshold=outlier_distance_threshold)),
                 ('Lasso', Lasso(alpha=alpha_lambda))]
        lasso = Pipeline(steps)

    lasso.fit(X, y)
    y_pred = lasso.predict(X)

    return np.sign(y_pred)


def get_legend_list():
    legend = []

    for threshold in defender_outlier_distance_thresholds:
        if threshold is not None:
            legend.append('T = {:d}'.format(threshold))
        else:
            legend.append('T = N/A')

    return legend


def main():
    num_aodt = len(attacker_outlier_distance_thresholds)
    num_dodt = len(defender_outlier_distance_thresholds)

    total_accuracy = np.zeros((num_aodt, num_dodt))
    outliers_classified = np.zeros((num_aodt, num_dodt))

    X_attack, y_attack = create_toy_data(random_state=0)
    yc = np.array([1])

    for j in range(num_aodt):
        print('fitting attack with T = {:.2f}'.format(attacker_outlier_distance_thresholds[j]))
        attack = fit_attack(X_attack, y_attack, outlier_distance_threshold=attacker_outlier_distance_thresholds[j])
        xc, yc = attack.get_attack_point(yc)

        for i in range(num_runs_error):
            X, y = create_toy_data(random_state=i+1)
            Xhat = np.append(X, xc, axis=0)
            yhat = np.append(y, yc, axis=0)

            for k in range(num_dodt):
                y_pred = fit_predict_defender(Xhat, yhat,
                                              outlier_distance_threshold=defender_outlier_distance_thresholds[k])

                nan_vals = np.isnan(y_pred)

                # count outliers detected
                if np.sum(nan_vals) != 0:
                    outliers_classified[j, k] = outliers_classified[j, k] + 1

                # remove outliers
                yhat_no_nan = yhat[~nan_vals]
                y_pred = y_pred[~nan_vals]

                total_accuracy[j, k] = total_accuracy[j, k] + accuracy_score(y_true=yhat_no_nan, y_pred=y_pred)

    classification_accuracy = total_accuracy/num_runs_error
    print(classification_accuracy)

    # save
    file_location = 'data/attack_strength_curve/'
    file_name_end = '_max{:.0f}_step{:.2f}.csv'.format(np.max(attacker_outlier_distance_thresholds),
                                                     attacker_outlier_distance_thresholds[1] -
                                                     attacker_outlier_distance_thresholds[0])
    np.savetxt('{:s}classification_accuracy{:s}'.format(file_location, file_name_end),
               np.vstack((attacker_outlier_distance_thresholds, classification_accuracy.T * 100)).T, delimiter=",")
    np.savetxt('{:s}outliers_classified{:s}'.format(file_location, file_name_end),
               outliers_classified, delimiter=",")

    # plot
    plt.figure(1)
    for i in range(num_dodt):
        plt.plot(attacker_outlier_distance_thresholds, classification_accuracy[:, i])

    plt.legend(get_legend_list(), loc='lower left')

    plt.ylabel('classification accuracy')
    plt.xlabel('attack strength')

    plt.figure(2)
    for i in range(num_dodt):
        plt.plot(attacker_outlier_distance_thresholds, outliers_classified[:, i])

    plt.legend(get_legend_list(), loc='lower left')

    plt.ylabel('num outliers found')
    plt.xlabel('attack strength')

    plt.show()


if __name__ == '__main__':
    main()
