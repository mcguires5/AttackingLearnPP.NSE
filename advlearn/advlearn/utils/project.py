import numpy as np
from copy import deepcopy


class Projector:
    def __init__(self, boundary=None):
        if boundary is not None:
            self.boundary = boundary

    def fit(self, data, projection="box"):
        if projection == "box":
            self.boundary = Projector.box_boundary(data)
        else:
            raise NotImplementedError("Invalid Projection Type")

    def is_out_of_bounds(self, data):
        return not np.array_equiv(data, self.project(data))

    def project(self, data):
        """Project X into the boundary region

        Parameters
        ----------
        data : input data

        Returns
        -------
        x_proj : projected data
        """
        data = np.ravel(data)
        x_proj = deepcopy(data)

        # Might need to replace 0th tile tuple with scalable variable eventually
        min_bound = np.tile(self.boundary[0, :], (1, 1)).flatten()
        max_bound = np.tile(self.boundary[1, :], (1, 1)).flatten()

        x_proj[data < self.boundary[0, :]] = min_bound[data < self.boundary[0, :]]
        x_proj[data > self.boundary[1, :]] = max_bound[data > self.boundary[1, :]]

        return x_proj

    @staticmethod
    def box_boundary(data):
        """Set the boundary based on the minimum and maximum of the input data
        """
        boundary = np.zeros((2, data.shape[1]))
        boundary[0, :] = np.amin(data, axis=0)
        boundary[1, :] = np.amax(data, axis=0)
        return boundary
