"""Visualization Package"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class PoisonViz(object):
    def __init__(
        self,
        attack_class,
        args,
        kwargs,
        data_training,
        labels_training,
        data_validation,
        labels_validation,
        verbose=2,
    ):
        self.attack_class = attack_class
        self.args = args
        self.kwargs = kwargs

        self.data_training = data_training
        self.labels_training = labels_training
        self.data_validation = data_validation
        self.labels_validation = labels_validation

        self.verbose = verbose  # 0 = Nothing, 1 = Minimal, 2 = All

        matplotlib.rcParams["axes.unicode_minus"] = False
        self.figure = plt.figure(figsize=(14, 8))

    def __fit_attack(self):
        attack = self.attack_class(*self.args, **self.kwargs)
        attack.fit(self.data_training, self.labels_training)
        return attack

    def __get_attack(self, n_points, target_class):
        attack = self.__fit_attack()
        x_attack, y_attack = attack.attack(n_points, target_class=target_class)
        return x_attack, y_attack

    def __get_trajectory(self, target):
        attack = self.__fit_attack()

        x_c = np.array([0, 1])
        y_c = np.array([target])

        x_attack_trajectory = attack.attack_trajectory(x_c, y_c)
        x_attack = x_attack_trajectory[-1, :]
        y_attack = y_c
        return x_attack_trajectory, x_attack, y_attack

    def __get_loss(self, xx, yy, xx_ravel, yy_ravel, target):
        attack = self.__fit_attack()

        if self.verbose >= 1:
            print("Calculating log loss background...\n")

        z_objective = np.zeros((xx.size,))
        for ind, (x, y) in enumerate(zip(xx_ravel, yy_ravel)):
            z_objective[ind] = attack.attack_loss(np.array([x, y]), np.array([target]))
        z_objective = z_objective.reshape(xx.shape)
        return z_objective

    def __get_gradient(self, xx, yy, xx_ravel, yy_ravel, target):
        attack = self.__fit_attack()

        if self.verbose >= 1:
            print("Calculating gradient mesh...\n")

        z_gradient = np.zeros((xx.size, 2))
        for ind, (x, y) in enumerate(zip(xx_ravel, yy_ravel)):
            z_gradient[ind, :] = attack.attack_direction(
                np.array([[x, y]]), np.array([target])
            )
        return z_gradient

    def plot_loss(
        self,
        subplot,
        mesh_x_steps,
        mesh_y_steps,
        gradient=True,
        decision=True,
        target=1,
    ):

        matplotlib.rc("pgf", rcfonts=False)
        matplotlib.rc("pgf", texsystem="pdflatex")

        ax = self.figure.add_subplot(*subplot)

        # Generate Meshgrid
        xx, yy = np.meshgrid(mesh_x_steps, mesh_y_steps)
        xx_ravel = xx.ravel()
        yy_ravel = yy.ravel()

        trajectory = False
        n_points = 1

        if trajectory:
            x_attack_trajectory, x_attack, y_attack = self.__get_trajectory(target)
        else:
            x_attack, y_attack = self.__get_attack(n_points, target)

        z_objective = self.__get_loss(xx, yy, xx_ravel, yy_ravel, target)
        if gradient:
            z_gradient = self.__get_gradient(xx, yy, xx_ravel, yy_ravel, target)

        # ax.set_title("Objective Function")

        cs_minor = ax.contourf(xx, yy, z_objective, 50, alpha=0.2, zorder=1)
        cs_major = ax.contour(xx, yy, z_objective, 5, alpha=1, zorder=2)
        ax.clabel(cs_major, colors="k", inline=True, fontsize=10, zorder=5)

        # Plot the Data
        ax.scatter(
            self.data_training[self.labels_training == -1, 0],
            self.data_training[self.labels_training == -1, 1],
            c="b",
            edgecolors="k",
            zorder=10,
        )
        ax.scatter(
            self.data_training[self.labels_training == 1, 0],
            self.data_training[self.labels_training == 1, 1],
            c="r",
            edgecolors="k",
            zorder=10,
        )

        # Plot the Trajectory and Attack Point
        """
        if trajectory:
            print('Drawing attack trajectory...')
            ax.plot(x_attack_trajectory[:, 0], x_attack_trajectory[:, 1],
                    c='r', zorder=20)
            ax.scatter(x_attack[0], x_attack[1], s=200, c='r',
                       marker='*', edgecolors='k', zorder=40)
        else:
            ax.scatter(x_attack[:, 0], x_attack[:, 1], s=200, c='r',
                       marker='*', edgecolors='k', zorder=40)
        """

        # Plot Boundary
        """
        ax.add_patch(patches.Rectangle(
            (x_boundary_min, y_boundary_min),
            x_boundary_max - x_boundary_min,
            y_boundary_max - y_boundary_min,
            fill=False,
            linestyle='dashed',
            zorder=30
        ))
        """

        # Plot the Gradient
        if gradient:
            mask = np.logical_or(z_gradient[:, 0] != 0, z_gradient[:, 1] != 0)
            if mask.any():
                print("Valid mask")
                ax.quiver(
                    xx_ravel[mask],
                    yy_ravel[mask],
                    z_gradient[mask, 0],
                    z_gradient[mask, 1],
                    alpha=0.2,
                )

        if decision:
            attack = self.__fit_attack()
            model = attack.fit_svm(self.data_training, self.labels_training)

            # Decision Boundary
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            z_decision = model.decision_function(xy).reshape(xx.shape)
            ax.contour(
                xx,
                yy,
                z_decision,
                colors="k",
                levels=[-1, 0, 1],
                alpha=0.5,
                linestyles=["--", "-", "--"],
            )

            # Support Vectors
            sv_ind = model.support_

            alpha = np.zeros_like(self.labels_training, dtype="float64")
            alpha[model.support_] = np.abs(model.dual_coef_.flatten())

            # margin_sv_ind = np.argwhere(np.logical_and(alpha > 1e-6, alpha < self.kwargs['c'] - 1e-6))
            margin_sv_ind = np.argwhere(
                np.logical_and(alpha > 0, alpha < self.kwargs["c"])
            )
            margin_sv_ind = margin_sv_ind.flatten()
            error_sv_ind = np.argwhere(alpha == self.kwargs["c"])
            error_sv_ind = error_sv_ind.flatten()
            # reserve_sv_ind = np.argwhere(alpha == 0)
            # reserve_sv_ind = reserve_sv_ind.flatten()

            score_raw = model.decision_function(self.data_training)
            score = np.multiply(score_raw, self.labels_training) - 1
            # score = np.multiply(score_raw, self.labels) - 2
            score = score / np.linalg.norm(score)

            score_size = score * 500

            # ax.scatter(self.data[score_size >= 0, 0], self.data[score_size >= 0, 1], s=score_size[score_size >= 0],
            #           facecolors='none', edgecolors='k', zorder=10)
            # ax.scatter(self.data[score_size < 0, 0], self.data[score_size < 0, 1], s=-score_size[score_size < 0],
            #           facecolors='none', edgecolors='r', zorder=10)

            # Plot All Support Vectors
            # ax.scatter(self.data[sv_ind, 0], self.data[sv_ind, 1], s=100, facecolors='none',
            #           edgecolors='k', zorder=10)

            ax.scatter(
                self.data_training[margin_sv_ind, 0],
                self.data_training[margin_sv_ind, 1],
                s=100,
                facecolors="none",
                edgecolors="red",
                zorder=15,
            )
            ax.scatter(
                self.data_training[error_sv_ind, 0],
                self.data_training[error_sv_ind, 1],
                s=100,
                facecolors="none",
                edgecolors="pink",
                zorder=15,
            )
            # ax.scatter(self.data[reserve_sv_ind, 0], self.data[reserve_sv_ind, 1], s=100, facecolors='none',
            #           edgecolors='purple', zorder=15)

        ax.set_autoscale_on(True)
        self.figure.colorbar(
            cs_minor, ax=ax, orientation="horizontal", format="%0.3f", pad=0.04
        )

    def show(self, package="matplotlib"):
        if package == "matplotlib":
            name = "viz"
            plt.savefig(name + ".pdf")
            plt.savefig(name + ".pgf")

            self.figure.show()
        elif package == "mpld3":
            pass
        else:
            raise ValueError("Invalid package")
