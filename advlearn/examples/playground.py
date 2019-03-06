def main():
    box_one()
    #box_two()

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
        X, y = make_blobs(n_samples=50, centers=centers, cluster_std=0.7,
                          n_features=2, random_state=random_state)
        #X, y = make_moons(n_samples=50, noise=0.5, random_state=random_state)
        y = (y * 2) - 1

        X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
        y = np.append(y, np.array([-1, -1, -1, 1]))
        return X, y

    def fit_attack(X, y):
        attack = LogisticAttack(boundary=boundary, step_size=1,
                                max_steps=5000, penalty='l2',
                                reg_inv=0.01, solver='saga',
                                congrad=False,
                                tol=1e-3, outlier_method=None,
                                outlier_weight=1,
                                outlier_distance_threshold=1,
                                outlier_power=2, outlier_k=3)
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
            z_gradient[idx, :] = attack.attack_direction(
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
            z_objective[idx] = attack.cross_entropy_error(
                np.array([x, y]), np.array([target]))
            idx += 1
        z_objective = z_objective.reshape(xx.shape)
        return z_objective

    def get_error():
        X, y = create_toy_data()
        attack = fit_attack(X,y)
        z_error = np.zeros((xx.size,))
        data_test, labels_test = create_toy_data()
        for j, _ in enumerate(xx_ravel):
            z_error[j] = attack.classification_error(
                np.array([xx_ravel[j], yy_ravel[j]]), np.array([target]),
                data_test, labels_test)
        z_error = z_error.reshape(xx.shape)
        return z_error

    def update_line(num, data, lines):
        for i in range(data.shape[1]):
            datas = data[:, i, :].T.reshape(data.shape[2], -1)
            lines[i].set_data(datas[..., :num])
        return lines

    def get_decision(attack=None):
        X, y = create_toy_data(random_state=0)
        if attack is None:
            attack = fit_attack(X, y)
        m, b = attack._fit_logistic_regression(X, y)
        t = np.linspace(-4, 4, 1000)
        boundary = -(t * m[0][0] + b[0]) / m[0][1]
        t = t[np.nonzero(np.abs(boundary) < 4)]
        boundary = boundary[np.nonzero(np.abs(boundary) < 4)]
        return t, boundary

    def get_final(xc, yc, attack=None):
        X, y = create_toy_data(random_state=0)
        data = np.vstack((X, xc))
        labels = np.hstack((y, yc))
        if attack is None:
            attack = fit_attack(data, labels)
        m, b = attack._fit_logistic_regression(X, y)
        t = np.linspace(-4, 4, 1000)
        boundary = -(t * m[0][0] + b[0]) / m[0][1]
        t = t[np.nonzero(np.abs(boundary) < 4)]
        boundary = boundary[np.nonzero(np.abs(boundary) < 4)]
        return t, boundary

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
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1],
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
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1],
                  alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f',
                    pad=0.04)

    plt.show()

def box_two():
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets.samples_generator import make_blobs
    from advlearn.attacks.poison import LinearAttack

    centers = [[-1, -0.5], [1, 0.5]]
    X, y = make_blobs(n_samples=50, centers=centers, cluster_std=1.0,
                      n_features=2, random_state=0)
    y = (y * 2) - 1

    X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
    y = np.append(y, np.array([-1, -1, -1, 1]))

    def fit_attack(X, y, method):
        # Generate Boundary
        x_boundary_min = -4
        x_boundary_max = 4
        y_boundary_min = -4
        y_boundary_max = 4
        boundary = np.array([[x_boundary_min, y_boundary_min],
                             [x_boundary_max, y_boundary_max]])

        attack = LinearAttack(boundary=boundary, lasso_lambda=0.02,
                              ridge_alpha=1, elasticnet_alpha=1,
                              elasticnet_l1_ratio=0.5,
                              efs_method='lasso', step_size=5,
                              max_steps=10000, outlier_method=method,
                              outlier_weight=1,
                              outlier_distance_threshold=1,
                              outlier_power=2, outlier_k=3,
                              surface_size=40)
        attack.fit(X, y)
        return attack

    methods = [None, 'distancethreshold']
    attacks = []

    for method in methods:
        attacks.append(fit_attack(X, y, method))

    points = 50
    nums = np.arange(1, points + 1)

    times = np.zeros((2, points))
    steps = np.zeros((2, points))

    assert nums.shape == times[0].shape, 'Shape mismatch'

    for a in range(len(attacks)):
        for n in nums:
            print('Attack:', a, '| Points:', n)
            start = time.time()
            xc, yc = attacks[a].get_attack_points(num_attack_points=n, yc=1,
                                                  plot=True)
            t = time.time() - start
            times[a][n - 1] = t
            steps[a][n - 1] = xc.shape[0] - 1
            print('Elapsed time:', t, 'seconds.\n')

    plt.subplot(1, 2, 1)
    plt.plot(nums, times[0], label='No outlier detection')
    plt.plot(nums, times[1], label='Distance threshold = 1')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of attack points generated')
    plt.ylabel('Elapsed time')
    plt.title('Elapsed time to converge on attack points')
    plt.subplot(1, 2, 2)
    plt.plot(nums, steps[0], label='No outlier detection')
    plt.plot(nums, steps[1], label='Distance threshold = 1')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of attack points generated')
    plt.ylabel('Number of steps')
    plt.title('Steps taken until convergence')

    plt.show()

def box_three():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as anim
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.datasets.samples_generator import make_moons
    from advlearn.attacks.poison.logistic_attack import LogisticAttack

    ##############
    # Parameters #
    ##############
    show_gradient = True

    x_mesh_min = -10.5
    x_mesh_max = 10.5
    y_mesh_min = -10.5
    y_mesh_max = 10.5
    h = 0.2  # step size in the mesh

    x_boundary_min = -10
    x_boundary_max = 10
    y_boundary_min = -10
    y_boundary_max = 10

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
        centers = [[-1, 0], [1, 0]]
        X, y = make_blobs(n_samples=50, centers=centers, cluster_std=0.7,
                          n_features=2, random_state=random_state)
        #X, y = make_moons(n_samples=50, noise=0.5, random_state=random_state)
        y = (y * 2) - 1

        X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
        y = np.append(y, np.array([-1, -1, -1, 1]))
        return X, y

    def fit_attack(X, y):
        attack = LogisticAttack(boundary=boundary, step_size=1,
                                max_steps=5000, penalty='l2',
                                reg_inv=0.01, solver='saga',
                                congrad=False,
                                tol=1e-3, outlier_method=None,
                                outlier_weight=1,
                                outlier_distance_threshold=1,
                                outlier_power=2, outlier_k=3)
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
            num_attack_points=q, yc=np.array([1]))
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
            z_gradient[idx, :] = attack.attack_direction(
                np.array([[x, y]]), np.array([1]))
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
            z_objective[idx] = attack.cross_entropy_error(
                np.array([x, y]), np.array([1]))
            idx += 1
        z_objective = z_objective.reshape(xx.shape)
        return z_objective

    def get_error():
        X, y = create_toy_data()
        attack = fit_attack(X,y)
        z_error = np.zeros((xx.size,))
        data_test, labels_test = create_toy_data()
        for j, _ in enumerate(xx_ravel):
            z_error[j] = attack.classification_error(
                np.array([xx_ravel[j], yy_ravel[j]]), np.array([1]),
                data_test, labels_test)
        z_error = z_error.reshape(xx.shape)
        return z_error

    def update_line(num, data, lines):
        for i in range(data.shape[1]):
            datas = data[:, i, :].T.reshape(data.shape[2], -1)
            lines[i].set_data(datas[..., :num])
        return lines

    def get_decision(attack=None):
        X, y = create_toy_data(random_state=0)
        if attack is None:
            attack = fit_attack(X, y)
        m, b = attack._fit_logistic_regression(X, y)
        t = np.linspace(-4, 4, 1000)
        boundary = -(t * m[0][0] + b[0]) / m[0][1]
        t = t[np.nonzero(np.abs(boundary) < 4)]
        boundary = boundary[np.nonzero(np.abs(boundary) < 4)]
        return t, boundary

    def get_final(xc, yc, attack=None):
        X, y = create_toy_data(random_state=0)
        data = np.vstack((X, xc))
        labels = np.hstack((y, yc))
        if attack is None:
            attack = fit_attack(data, labels)
        m, b = attack._fit_logistic_regression(X, y)
        t = np.linspace(-4, 4, 1000)
        boundary = -(t * m[0][0] + b[0]) / m[0][1]
        t = t[np.nonzero(np.abs(boundary) < 4)]
        boundary = boundary[np.nonzero(np.abs(boundary) < 4)]
        return t, boundary

    # Get Data and Trajectory
    X, y = create_toy_data(random_state=0)
    x_attack_trajectory, x_attack, y_attack = get_trajectory()
    t, b = get_decision()
    ft, fb = get_final(x_attack, y_attack)

    if show_gradient:
        z_gradient = get_gradient()

    # Plot the data
    matplotlib.rcParams['axes.unicode_minus'] = False
    figure = plt.figure(figsize=(14, 8))

    ###########################
    # Plot Objective Function #
    ###########################
    # Plot Objective Function Background
    z_objective = get_objective()
    ax3d = figure.add_subplot(2, 2, 3, projection='3d')
    ax3d.plot_surface(xx, yy, z_objective, cmap='viridis')

    ax = plt.subplot(2, 2, 1)
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

    # Plot decision boundary
    ax.plot(t, b)
    ax.plot(ft, fb)

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
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1],
                  alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f',
                    pad=0.04)

    #############################
    # Plot Classification Error #
    #############################
    # Plot Classification Error Background
    z_error_full = np.zeros(xx.shape + (num_runs_error,))
    for j in range(num_runs_error):
        z_error_full[:, :, j] = get_error()
    z_error = np.mean(z_error_full, axis=2)

    ax3d = figure.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot_surface(xx, yy, z_error, cmap='viridis')

    ax = plt.subplot(2, 2, 2)
    ax.set_title("Classification Error")

    cs_minor = ax.contourf(xx, yy, z_error, 20, alpha=.2, zorder=1)
    cs_major = ax.contour(xx, yy, z_error, 5, alpha=1, zorder=2)
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

    # Plot decision boundary
    ax.plot(t, b)
    ax.plot(ft, fb)

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
        ax.quiver(xx_ravel, yy_ravel, z_gradient[:, 0], z_gradient[:, 1],
                  alpha=0.2)

    ax.set_autoscale_on(True)
    figure.colorbar(cs_minor, ax=ax, orientation='horizontal', format='%0.3f',
                    pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()