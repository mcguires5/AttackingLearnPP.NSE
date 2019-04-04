"""Visualization Package"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn import metrics


class ComparisonViz(object):
    def __init__(
        self,
        attack_classes,
        data_training,
        labels_training,
        data_validation,
        labels_validation,
        results=None,
        verbose=2,
    ):
        self.attack_classes = attack_classes

        self.data_training = data_training
        self.labels_training = labels_training
        self.data_validation = data_validation
        self.labels_validation = labels_validation

        self.verbose = verbose  # 0 = Nothing, 1 = Minimal, 2 = All

        matplotlib.rcParams["axes.unicode_minus"] = False
        self.figure = None
        self.results = results
        self.results_attack_x = None
        self.results_attack_y = None

    def __fit_attack(self, key):
        model = self.attack_classes[key]["model"]
        args = self.attack_classes[key]["args"]
        kwargs = self.attack_classes[key]["kwargs"]

        attack = model(*args, **kwargs)
        attack.fit(self.data_training, self.labels_training)
        return attack

    def __get_attack(self, key, n_points, target_class):
        attack = self.__fit_attack(key)
        x_attack, y_attack = attack.attack(n_points, target_class=target_class)
        return x_attack, y_attack

    def __get_all_attacks(self, n_points, target_class):
        x_results = np.zeros((len(self.attack_classes), self.data_training.shape[1]))
        y_results = np.zeros((len(self.attack_classes)))
        for ind_alg, (key, algorithm) in enumerate(self.attack_classes.items()):
            print("Running attack: {}".format(key))
            # Generate the Attack
            x_attack, y_attack = self.__get_attack(key, n_points, target_class)
            x_results[ind_alg, :] = x_attack
            y_results[ind_alg] = y_attack
        return x_results, y_results

    def __get_loss(self, *args, target=-1):

        # Generate Meshgrid
        grid = np.meshgrid(*args)
        n_dim = len(grid)
        n_points = grid[0].size
        attack_points = np.zeros((n_points, n_dim))
        for ind_dimm, dim in enumerate(grid):
            attack_points[:, ind_dimm] = grid[ind_dimm].ravel()

        if self.verbose >= 1:
            print("Calculating log loss background...\n")

        results = np.zeros((len(self.attack_classes), n_points))
        for ind_alg, (key, algorithm) in enumerate(self.attack_classes.items()):
            print("Running attack: {}".format(key))
            # Fit the Attack
            attack = self.__fit_attack(key)
            # Calculate the Loss
            for ind_loss, attack_point in enumerate(attack_points):
                results[ind_alg, ind_loss] = attack.attack_loss(
                    attack_point, np.array([target])
                )

        results = results - np.amin(results, axis=1)[:, None]

        inds_to_replace = np.logical_not(np.any(results, axis=1))
        results[inds_to_replace, :] = 1

        results = results / np.sum(results, axis=1)[:, None]
        return results

    def plot_comparison(
        self, *args, distance="euclidean", comparison_type="full", target=1
    ):

        matplotlib.rc("pgf", rcfonts=False)
        matplotlib.rc("pgf", texsystem="pdflatex")

        if comparison_type == "full":
            # Get the loss results if you have not already
            if self.results is None:
                self.results = self.__get_loss(*args, target=target)

            if distance == "hellinger":
                distances = cdist(
                    self.results,
                    self.results,
                    lambda p, q: np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
                    / np.sqrt(2),
                )
            else:
                distances = cdist(self.results, self.results, distance)

            if np.isnan(distances).any():
                print("WARNING! Replacing nan with zero")
                distances = np.nan_to_num(distances)

        elif comparison_type == "max":
            # Get the loss results if you have not already
            if self.results is None:
                self.results = self.__get_loss(*args, target=target)

            distances = np.zeros((self.results.shape[0], self.results.shape[0]))

            max_loss = np.amax(self.results, axis=1)
            for ind_from in range(self.results.shape[0]):
                from_loss_ind = np.argmax(self.results[ind_from])
                for ind_to in range(self.results.shape[0]):
                    to_loss = self.results[ind_to, from_loss_ind]
                    distances[ind_from, ind_to] = max_loss[ind_to] - to_loss

        elif comparison_type == "real":
            # Get the attack results if you have not already
            if self.results_attack_x is None or self.results_attack_y is None:
                self.results_attack_x, self.results_attack_y = self.__get_all_attacks(
                    1, target
                )

            distances = np.zeros(
                (self.results_attack_x.shape[0], self.results_attack_x.shape[0])
            )
            for ind_from, (key_from, algorithm_from) in enumerate(
                self.attack_classes.items()
            ):
                poisoned_x = np.append(
                    self.data_training,
                    self.results_attack_x[ind_from, :].reshape(1, -1),
                    axis=0,
                )
                poisoned_y = np.append(
                    self.labels_training,
                    self.results_attack_y[ind_from].reshape(1),
                    axis=0,
                )

                for ind_to, (key_to, algorithm_to) in enumerate(
                    self.attack_classes.items()
                ):
                    attack_to = algorithm_to["model"](
                        *algorithm_to["args"], **algorithm_to["kwargs"]
                    )
                    model_to = attack_to.surrogate_model()
                    model_to.fit(poisoned_x, poisoned_y)
                    distances[ind_from, ind_to] = model_to.score(
                        self.data_validation, self.labels_validation
                    )
            distances = (
                100
                * (np.diag(distances).reshape(1, -1) - distances)
                / ((np.diag(distances).reshape(1, -1) + distances) / 2)
            )
        else:
            raise ValueError()

        df = pd.DataFrame(
            distances, self.attack_classes.keys(), self.attack_classes.keys()
        )

        # current_palette = sns.color_palette()
        # row_values = np.mean(df.values, axis=0)
        # col_values = np.mean(df.values, axis=1)
        # lut = dict(zip(set(row_values), sns.hls_palette(len(set(row_values)), l=0.5, s=0.8)))
        # row_colors = row_values.map(lut)

        # https://seaborn.pydata.org/examples/structured_heatmap.html
        grid = sns.clustermap(
            df, center=0, cmap="vlag", linewidths=0.75, figsize=(11, 7)
        )
        plt.setp(grid.ax_heatmap.get_yticklabels(), rotation=0)
        plt.title("Poisoning Transferability")
        plt.xticks(rotation=45)

        name = "transferability_" + comparison_type + "_" + distance
        plt.savefig(name + ".pdf")
        plt.savefig(name + ".pgf")

        plt.show()

    def save_comparison(
        self, *args, distance="euclidean", comparison_type="full", target=1, key=1
    ):

        if comparison_type == "full":
            # Get the loss results if you have not already
            if self.results is None:
                self.results = self.__get_loss(*args, target=target)

            if distance == "hellinger":
                distances = cdist(
                    self.results,
                    self.results,
                    lambda p, q: np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
                    / np.sqrt(2),
                )
            else:
                distances = cdist(self.results, self.results, distance)

            if np.isnan(distances).any():
                print("WARNING! Replacing nan with zero")
                distances = np.nan_to_num(distances)

        elif comparison_type == "max":
            # Get the loss results if you have not already
            if self.results is None:
                self.results = self.__get_loss(*args, target=target)

            distances = np.zeros((self.results.shape[0], self.results.shape[0]))

            max_loss = np.amax(self.results, axis=1)
            for ind_from in range(self.results.shape[0]):
                from_loss_ind = np.argmax(self.results[ind_from])
                for ind_to in range(self.results.shape[0]):
                    to_loss = self.results[ind_to, from_loss_ind]
                    distances[ind_from, ind_to] = max_loss[ind_to] - to_loss

        elif comparison_type == "real":
            # Get the attack results if you have not already
            if self.results_attack_x is None or self.results_attack_y is None:
                self.results_attack_x, self.results_attack_y = self.__get_all_attacks(
                    1, target
                )

            distances = np.zeros(
                (self.results_attack_x.shape[0], self.results_attack_x.shape[0])
            )
            for ind_from, (key_from, algorithm_from) in enumerate(
                self.attack_classes.items()
            ):
                poisoned_x = np.append(
                    self.data_training,
                    self.results_attack_x[ind_from, :].reshape(1, -1),
                    axis=0,
                )
                poisoned_y = np.append(
                    self.labels_training,
                    self.results_attack_y[ind_from].reshape(1),
                    axis=0,
                )

                for ind_to, (key_to, algorithm_to) in enumerate(
                    self.attack_classes.items()
                ):
                    attack_to = algorithm_to["model"](
                        *algorithm_to["args"], **algorithm_to["kwargs"]
                    )
                    model_to = attack_to.surrogate_model()
                    model_to.fit(poisoned_x, poisoned_y)
                    distances[ind_from, ind_to] = model_to.score(
                        self.data_validation, self.labels_validation
                    )
            distances = (
                100
                * (np.diag(distances).reshape(1, -1) - distances)
                / ((np.diag(distances).reshape(1, -1) + distances) / 2)
            )
        else:
            raise ValueError()

        df = pd.DataFrame(
            distances, self.attack_classes.keys(), self.attack_classes.keys()
        )
        name = (
            "transferability_"
            + comparison_type
            + "_"
            + distance
            + "_"
            + str(key)
            + ".pkl"
        )
        df.to_pickle(name)
