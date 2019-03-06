"""Pipeline"""

import numpy as np


class Pipeline(object):
    """Pipeline"""

    def __init__(self, steps):
        self.steps = steps

    def predict(self, X):
        """Apply the transformations and predict the final estimator"""
        x_step = X
        result_used = np.ones((X.shape[0],), dtype=bool)  # All data is being used, so set everything to true

        for name, step in self.steps[:-1]:
            x_step, used = step.transform(x_step)
            used_index = np.where(result_used)[0][used]  # Map used to the currently used data
            result_used = np.zeros((X.shape[0],), dtype=bool)  # Set everything to false
            result_used[used_index] = True  # Set the currently used data to true

        result = np.empty((X.shape[0],))
        result[:] = np.nan
        result[result_used] = self._final_estimator.predict(x_step)
        return result

    def fit(self, X, y):
        """Apply the transformations and fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            x_step, y_step, used = step.transform(x_step, y_step)
        self._final_estimator.fit(x_step, y_step)
        return

    def partial_fit(self, X, y):
        """Apply the transformations and then partial fit the final estimator"""
        x_step = X
        y_step = y
        for name, step in self.steps[:-1]:
            x_step, y_step, used = step.transform(x_step, y_step)
        self._final_estimator.partial_fit(x_step, y_step)
        return

    @property
    def _final_estimator(self):
        return self.steps[-1][-1]
