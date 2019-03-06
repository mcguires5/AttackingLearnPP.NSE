"""Generate Toy Gaussian Objective Function Figure"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets.samples_generator import make_blobs
from advlearn.attacks.poison import GeneralTikhonovAttack


##############
# Parameters #
##############
show_gradient = False
x_mesh_min = -4.5
x_mesh_max = 4.5
y_mesh_min = -4.5
y_mesh_max = 4.5
h = 0.2  # step size in the mesh

x_boundary_min = -4
x_boundary_max = 4
y_boundary_min = -4
y_boundary_max = 4

num_runs_error = 5

# Generate Meshgrid
xx, yy = np.meshgrid(np.arange(x_mesh_min, x_mesh_max, h),
                     np.arange(y_mesh_min, y_mesh_max, h))

xx_ravel = xx.ravel()
yy_ravel = yy.ravel()

# Generate Boundary
boundary = np.array([[x_boundary_min, y_boundary_min], [x_boundary_max, y_boundary_max]])


def create_toy_data(random_state=None):
    centers = [[-1, 0], [1, 0]]
    X, y = make_blobs(n_samples=50, centers=centers, cluster_std=0.5, n_features=2, random_state=random_state)
    y = (y * 2) - 1

    X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
    y = np.append(y, np.array([-1, -1, -1, 1]))
    return X, y


def fit_attack(X, y):
    attack = GeneralTikhonovAttack(step_size=5, max_steps=5000,
                                   lasso_lambda=0.02,
                                   boundary=boundary,
                                   outlier_method='distancethreshold',
                                   outlier_distance_threshold=1)
    attack.fit(X, y)
    return attack


def get_trajectory(attack=None):
    if attack is None:
        X, y = create_toy_data(random_state=0)
        attack = fit_attack(X, y)
    x_attack_trajectory, y_attack = attack.get_attack_point_trajectory(yc=np.array([1]))
    x_attack = np.reshape(x_attack_trajectory[-1, :], (1, -1))
    return x_attack_trajectory, x_attack, y_attack


def get_gradient(attack=None):
    if attack is None:
        X, y = create_toy_data(random_state=0)
        attack = fit_attack(X, y)
    z_gradient = np.zeros((xx.size, 2))
    for j, _ in enumerate(xx_ravel):
        z_gradient[j, :] = attack.attacker_objective_direction(np.array([xx_ravel[j], yy_ravel[j]]), np.array([1]))
    return z_gradient


def get_objective(attack=None):
    if attack is None:
        X, y = create_toy_data(random_state=0)
        attack = fit_attack(X, y)
    z_objective = np.zeros((xx.size,))
    for j, _ in enumerate(xx_ravel):
        z_objective[j] = attack.mse_error(np.array([xx_ravel[j], yy_ravel[j]]), np.array([1]))
    z_objective = z_objective.reshape(xx.shape)
    return z_objective


def get_error():
    X, y = create_toy_data()
    attack = fit_attack(X, y)
    z_error = np.zeros((xx.size,))
    for j, _ in enumerate(xx_ravel):
        z_error[j] = attack.classification_error(np.array([xx_ravel[j], yy_ravel[j]]), np.array([1]))
    z_error = z_error.reshape(xx.shape)
    return z_error


def main():
    # Get Data and Trajectory
    X, y = create_toy_data(random_state=0)
    x_attack_trajectory, x_attack, y_attack = get_trajectory()

    if show_gradient:
        z_gradient = get_gradient()

    # Plot the data
    matplotlib.rcParams['axes.unicode_minus'] = False
    figure = plt.figure(figsize=(14, 8))

    ###########################
    # Plot Objective Function #
    ###########################
    figure = plt.figure(figsize=(7, 8))
    ax = plt.axes()

    # Plot Objective Function Background
    z_objective = get_objective()

    cs_minor = ax.contourf(xx, yy, z_objective, 50, alpha=.2, zorder=1)
    cs_major = ax.contour(xx, yy, z_objective, 5, alpha=1, zorder=2)
    ax.clabel(cs_major, colors='k', inline=True, fontsize=10, zorder=5)

    # Plot the Data
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='b', edgecolors='k', zorder=10)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='r', edgecolors='k', zorder=10)

    # Plot the Trajectory and Attack Point
    ax.plot(x_attack_trajectory[:, 0], x_attack_trajectory[:, 1], c='r', zorder=20)
    ax.scatter(x_attack_trajectory[0, 0], x_attack_trajectory[0, 1], s=75, c='r', edgecolors='k', zorder=40)
    ax.scatter(x_attack[:, 0], x_attack[:, 1], s=200, c='r', marker='*', edgecolors='k', zorder=40)

    # Plot Boundary
    ax.add_patch(patches.Rectangle(
        (x_boundary_min, y_boundary_min),
        x_boundary_max - x_boundary_min,
        y_boundary_max - y_boundary_min,
        fill=False,
        linestyle='dashed',
        zorder=30
    ))

    # Plot the Gradient
    if show_gradient:
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1], alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f', pad=0.04)

    plt.savefig('gaussian_toy_distance_threshold_objective.pdf')
    plt.savefig('gaussian_toy_distance_threshold_objective.eps')

    #############################
    # Plot Classification Error #
    #############################
    figure = plt.figure(figsize=(7, 8))
    ax = plt.axes()

    # Plot Classification Error Background
    z_error_full = np.zeros(xx.shape + (num_runs_error,))
    for j in range(num_runs_error):
        z_error_full[:, :, j] = get_error()
    z_error = np.mean(z_error_full, axis=2)

    cs_minor = ax.contourf(xx, yy, z_error, 20, alpha=.2, zorder=1)
    cs_major = ax.contour(xx, yy, z_error, 5, alpha=1, zorder=2)
    ax.clabel(cs_major, colors='k', inline=True, fontsize=10, zorder=5)

    # Plot the Data
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='b', edgecolors='k', zorder=10)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='r', edgecolors='k', zorder=10)

    # Plot the Trajectory and Attack Point
    ax.plot(x_attack_trajectory[:, 0], x_attack_trajectory[:, 1], c='r', zorder=20)
    ax.scatter(x_attack_trajectory[0, 0], x_attack_trajectory[0, 1], s=75, c='r', edgecolors='k', zorder=40)
    ax.scatter(x_attack[:, 0], x_attack[:, 1], s=200, c='r', marker='*', edgecolors='k', zorder=40)

    # Plot Boundary
    ax.add_patch(patches.Rectangle(
        (x_boundary_min, y_boundary_min),
        x_boundary_max - x_boundary_min,
        y_boundary_max - y_boundary_min,
        fill=False,
        linestyle='dashed',
        zorder=30
    ))

    # Plot the Gradient
    if show_gradient:
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1], alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f', pad=0.04)

    plt.savefig('gaussian_toy_distance_threshold_classification.pdf')
    plt.savefig('gaussian_toy_distance_threshold_classification.eps')


if __name__ == '__main__':
    main()
