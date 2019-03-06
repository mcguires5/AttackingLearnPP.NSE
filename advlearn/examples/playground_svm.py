def main():
    box_one()

def box_one():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as anim
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.datasets.samples_generator import make_moons
    from advlearn.attacks.poison.logistic_attack import LogisticAttack
    from advlearn.attacks.poison import SVMAttack

    ##############
    # Parameters #
    ##############
    target = 1
    show_gradient = True

    mesh = 10.5
    x_mesh_min = -mesh
    x_mesh_max = mesh
    y_mesh_min = -mesh
    y_mesh_max = mesh

    h = 0.5  # step size in the mesh

    bound = mesh-0.5
    x_boundary_min = -bound
    x_boundary_max = bound
    y_boundary_min = -bound
    y_boundary_max = bound

    num_runs_error = 50

    # Generate Meshgrid
    xx, yy = np.meshgrid(np.arange(x_mesh_min, x_mesh_max, h),
                         np.arange(y_mesh_min, y_mesh_max, h))

    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()

    # Generate Boundary
    boundary = np.array([[x_boundary_min, y_boundary_min],
                         [x_boundary_max, y_boundary_max]])

    def create_toy_data(random_state=None):
        centers = [[-1, -0.7], [1, 0.7]]
        X, y = make_blobs(n_samples=100, centers=centers, cluster_std=0.5,
                          n_features=2, random_state=random_state)
        #X, y = make_moons(n_samples=50, noise=0.5, random_state=random_state)
        y = (y * 2) - 1

        X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
        y = np.append(y, np.array([-1, -1, -1, 1]))
        return X, y

    def fit_attack(X, y):
        attack = SVMAttack(boundary=boundary, step_size=0.01, max_steps=5000,
                           c=50, kernel='linear', degree=3, gamma=0.5)
        attack.fit(X, y)
        return attack

    def get_trajectory(attack=None):
        if attack is None:
            X, y = create_toy_data(random_state=0)
            attack = fit_attack(X, y)
        import time
        q = 1
        start = time.time()
        x_attack_trajectory, y_attack = attack.get_attack_point_trajectory(
            num_attack_points=q, yc=np.array([target]))
        print('Elapsed time:', time.time() - start, 'seconds.\n')
        x_attack = np.reshape(x_attack_trajectory[-q, :], (q, -1))
        return x_attack_trajectory, x_attack, y_attack

    def get_gradient(attack=None):
        if attack is None:
            X, y = create_toy_data(random_state=0)
            attack = fit_attack(X, y)
        print('Calculating gradient mesh...\n')
        z_gradient = np.zeros((xx.size, 2))
        idx = 0
        for x, y in zip(xx_ravel, yy_ravel):
            z_gradient[idx, :] = attack.attacker_objective_direction(
                np.array([[x, y]]), np.array([target]))
            idx += 1
        return z_gradient

    def get_objective(attack=None):
        if attack is None:
            X, y = create_toy_data(random_state=0)
            attack = fit_attack(X, y)
        print('Calculating log loss background...\n')
        z_objective = np.zeros((xx.size,))
        idx = 0
        for x, y in zip(xx_ravel, yy_ravel):
            z_objective[idx] = attack.error(
                np.array([x, y]), np.array([target]))
            idx += 1
        z_objective = z_objective.reshape(xx.shape)
        return z_objective

    def update_line(num, data, lines):
        for i in range(data.shape[1]):
            datas = data[:, i, :].T.reshape(data.shape[2], -1)
            lines[i].set_data(datas[..., :num])
        return lines

    # Get Data and Trajectory
    X, y = create_toy_data(random_state=0)
    x_attack_trajectory, x_attack, y_attack = get_trajectory()

    if show_gradient:
        z_gradient = get_gradient()

    # Plot the data
    matplotlib.rcParams['axes.unicode_minus'] = False
    figure = plt.figure(figsize=(14, 8))

    ########################################
    # Plot Objective Function for Target 1 #
    ########################################
    # Plot Objective Function Background
    z_objective = get_objective()

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Objective Function")

    cs_minor = ax.contourf(xx, yy, z_objective, 50, alpha=.2, zorder=1)
    cs_major = ax.contour(xx, yy, z_objective, 5, alpha=1, zorder=2)
    ax.clabel(cs_major, colors='k', inline=True, fontsize=10, zorder=5)

    # Plot the Data
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='b', edgecolors='k', zorder=10)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='r', edgecolors='k', zorder=10)

    # Plot the Trajectory and Attack Point
    print('Drawing attack trajectory...')
    for i in range(x_attack_trajectory.shape[1]):
        ax.plot(x_attack_trajectory[:, i, 0], x_attack_trajectory[:, i, 1],
                c='r', zorder=20)
        ax.scatter(x_attack_trajectory[0, i, 0], x_attack_trajectory[0, i, 1],
                   s=75, c='r', edgecolors='k', zorder=40)
    ax.scatter(x_attack[:, 0], x_attack[:, 1], s=200, c='r',
               marker='*', edgecolors='k', zorder=40)

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
        mask = np.logical_or(z_gradient[:, 0] != 0, z_gradient[:, 1] != 0)
        if mask.any():
            print('Valid mask')
            ax.quiver(xx_ravel[mask], yy_ravel[mask],
                      z_gradient[mask, 0], z_gradient[mask, 1],
                      alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f',
                    pad=0.04)

    #########################################
    # Plot Objective Function for Target -1 #
    #########################################
    target = -1

    # Get Data and Trajectory
    X, y = create_toy_data(random_state=0)
    x_attack_trajectory, x_attack, y_attack = get_trajectory()

    if show_gradient:
        z_gradient = get_gradient()

    # Plot Objective Function Background
    z_objective = get_objective()

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Objective Function")

    cs_minor = ax.contourf(xx, yy, z_objective, 50, alpha=.2, zorder=1)
    cs_major = ax.contour(xx, yy, z_objective, 5, alpha=1, zorder=2)
    ax.clabel(cs_major, colors='k', inline=True, fontsize=10, zorder=5)

    # Plot the Data
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='b', edgecolors='k', zorder=10)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='r', edgecolors='k', zorder=10)

    # Plot the Trajectory and Attack Point
    print('Drawing attack trajectory...')
    for i in range(x_attack_trajectory.shape[1]):
        ax.plot(x_attack_trajectory[:, i, 0], x_attack_trajectory[:, i, 1],
                c='r', zorder=20)
        ax.scatter(x_attack_trajectory[0, i, 0], x_attack_trajectory[0, i, 1],
                   s=75, c='r', edgecolors='k', zorder=40)
    ax.scatter(x_attack[:, 0], x_attack[:, 1], s=200, c='r',
               marker='*', edgecolors='k', zorder=40)

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
        mask = np.logical_or(z_gradient[:, 0] != 0, z_gradient[:, 1] != 0)
        if mask.any():
            print('Valid mask')
            ax.quiver(xx_ravel[mask], yy_ravel[mask],
                      z_gradient[mask, 0], z_gradient[mask, 1],
                      alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f',
                    pad=0.04)

    plt.show()

if __name__ == '__main__':
    main()