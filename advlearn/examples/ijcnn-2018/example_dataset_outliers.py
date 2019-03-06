"""Test that the real data are outliers"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets.samples_generator import make_blobs

from advlearn.attacks.poison import GeneralTikhonovAttack

dataset_name = 'spambase'
num_d = 8

spambase_train = np.loadtxt('csv/spambase_train.csv', delimiter=',')
credit_approval_train = np.loadtxt('csv/credit-approval_train.csv', delimiter=',')
congressional_voting_train = np.loadtxt('csv/congressional-voting_train.csv', delimiter=',')

spambase_test = np.loadtxt('csv/spambase_train.csv', delimiter=',')
credit_approval_test = np.loadtxt('csv/credit-approval_test.csv', delimiter=',')
congressional_voting_test = np.loadtxt('csv/congressional-voting_test.csv', delimiter=',')

if dataset_name is 'spambase':
    data_train_x = spambase_train[:, :-2]
    data_train_y = (spambase_train[:, -1] * 2) - 1
    data_test_x = spambase_test[:, :-2]
    data_test_y = (spambase_test[:, -1] * 2) - 1
    step_size = 5000
    max_steps = 1000
    boundary_min = 0
    boundary_max = 20
elif dataset_name is 'credit-approval':
    data_train_x = credit_approval_train[:, :-2]
    data_train_y = (credit_approval_train[:, -1] * 2) - 1
    data_test_x = credit_approval_test[:, :-2]
    data_test_y = (credit_approval_test[:, -1] * 2) - 1
    step_size = 500
    max_steps = 1000
    boundary_min = -20
    boundary_max = 20
elif dataset_name is 'congressional-voting':
    data_train_x = congressional_voting_train[:, :-2]
    data_train_y = (congressional_voting_train[:, -1] * 2) - 1
    data_test_x = congressional_voting_test[:, :-2]
    data_test_y = (congressional_voting_test[:, -1] * 2) - 1
    step_size = 500
    max_steps = 1000
    boundary_min = -20
    boundary_max = 20

boundary = np.concatenate((boundary_min * np.ones((1, data_test_x.shape[1])),
                           boundary_max * np.ones((1, data_test_x.shape[1]))))
attack = GeneralTikhonovAttack(boundary=boundary, step_size=step_size, max_steps=max_steps)
attack.fit(data_test_x, data_test_y)
attack_x, attack_y = attack.get_attack_point_trajectory()

print(attack_x[-1, :])

plt.figure(figsize=(12, 11))
for i in range(num_d):
    for j in range(num_d):
        print((i * num_d) + j + 1)
        subplot = plt.subplot(num_d, num_d, (i * num_d) + j + 1)
        if i == j:
            plt.hist(data_test_x[:, i])
        else:
            plt_color = np.where(data_test_y == 1, 'b', 'r')
            plt.scatter(data_test_x[:, i], data_test_x[:, j], color=plt_color)
            plt.plot(attack_x[:, i], attack_x[:, j], color='r')
            plt.scatter(attack_x[-1, i], attack_x[-1, j], marker='*', s=100, color='r')

plt.show()
