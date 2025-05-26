from typing import Optional

import numpy as np
from poly_matrix.least_squares_problem import LeastSquaresProblem
from poly_matrix.poly_matrix import PolyMatrix

from popr.base_lifters import StateLifter


class Stereo1DLifter(StateLifter):
    """Toy example for stereo localization in 1D. We minimize the following cost function:

    .. math::
        f(\\theta) = \\sum_{j=0}^{N-1} (u_j - 1 / (\\theta - a_j))^2

    where :math:`a_j` are the landmarks and :math:`u_j` are the measurements.

    This is the pedagogical running example of `this paper <https://arxiv.org/abs/2308.05783>`_.
    and also used in the :ref:`Quick Start Guide`.
    """

    PARAM_LEVELS = ["no", "p", "ppT"]
    VARIABLE_LIST = [["h", "x"], ["h", "x", "z_0"], ["h", "x", "z_0", "z_1"]]

    NOISE = 0.1

    def __init__(self, n_landmarks, param_level="no"):
        self.n_landmarks = n_landmarks
        self.d = 1
        self.W = 1.0

        # will be initialized later
        self.landmarks_ = None

        super().__init__(param_level=param_level, d=self.d, n_parameters=n_landmarks)

    @property
    def landmarks(self):
        if self.landmarks_ is None:
            self.landmarks_ = np.random.rand(self.n_landmarks, self.d)
        return self.landmarks_

    def sample_parameters(self, theta=None):
        if self.parameters_ is None:
            return self.sample_parameters_landmarks(self.landmarks)
        landmarks = np.random.rand(self.n_landmarks, self.d)
        return self.sample_parameters_landmarks(landmarks)

    def sample_theta(self):
        x_try = np.random.rand(1)
        counter = 0
        while np.min(np.abs(x_try - self.landmarks)) <= 1e-2:
            x_try = np.random.rand(1)
            counter += 1
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        return x_try

    def get_x(self, theta=None, parameters=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        if var_subset is None:
            var_subset = self.var_dict.keys()

        if self.param_level == "no":
            landmarks = {
                f"p_{i}": self.landmarks[i] for i in range(self.landmarks.shape[0])
            }
        else:
            landmarks = {
                f"p_{i}": parameters[f"p_{i}"][: self.d]
                for i in range(self.landmarks.shape[0])
            }

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif key == "x":
                x_data.append(theta[0])
            elif "z" in key:
                idx = int(key.split("_")[-1])
                x_data.append(1 / (theta[0] - landmarks[f"p_{idx}"]))
            else:
                raise ValueError("unknown key in get_x", key)
        return np.hstack(x_data)

    @property
    def var_dict(self):
        vars = [self.HOM, "x"] + [f"z_{j}" for j in range(self.n_landmarks)]
        return {v: 1 for v in vars}

    @property
    def param_dict(self):
        return self.param_dict_landmarks

    def get_Q(self, noise: Optional[float] = None):
        if self.landmarks is None:
            raise ValueError("self.landmarks must be initialized before calling get_Q.")
        if noise is None:
            noise = self.NOISE

        y = 1 / (self.theta - self.landmarks.flatten()) + np.random.normal(
            scale=noise, loc=0, size=self.n_landmarks
        )
        if self.y_ is None:
            self.y_ = y

        return self.get_Q_from_y(y)

    def get_Q_from_y(self, y):
        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({self.HOM: -y[j], f"z_{j}": 1})
        return ls_problem.get_Q().get_matrix(self.var_dict)

    def get_A_known(self, var_dict=None, output_poly=False):
        if var_dict is None:
            var_dict = self.var_dict

        # if self.add_parameters:
        #    raise ValueError("can't extract known matrices yet when using parameters.")

        A_known = []

        # enforce that z_j = 1/(x - a_j) <=> 1 - z_j*x + a_j*z_j = 0
        if not ("x" in var_dict and self.HOM in var_dict):
            return []

        landmark_indices = [
            int(key.split("_")[-1]) for key in var_dict if key.startswith("z_")
        ]
        for j in landmark_indices:
            A = PolyMatrix()
            A[self.HOM, f"z_{j}"] = 0.5 * self.landmarks[j]
            A["x", f"z_{j}"] = -0.5
            A[self.HOM, self.HOM] = 1.0
            if output_poly:
                A_known.append(A)
            else:
                A_known.append(A.get_matrix(variables=self.var_dict))
        return A_known

    def get_A_known_redundant(self, var_dict=None, output_poly=False):
        import itertools

        if var_dict is None:
            var_dict = self.var_dict

        assert self.HOM in var_dict, "homogenization variable must be in var_dict"

        landmark_indices = [
            int(key.split("_")[-1]) for key in var_dict if key.startswith("z_")
        ]
        # add known redundant constraints:
        # enforce that z_j - z_i = (a_j - a_i) * z_j * z_i
        A_known = []
        for i, j in itertools.combinations(landmark_indices, 2):
            A = PolyMatrix()
            A[self.HOM, f"z_{j}"] = 1
            A[self.HOM, f"z_{i}"] = -1
            A[f"z_{i}", f"z_{j}"] = self.landmarks[i] - self.landmarks[j]
            if output_poly:
                A_known.append(A)
            else:
                A_known.append(A.get_matrix(variables=self.var_dict))
        return A_known

    def get_cost(self, t, y):
        return np.sum((y - (1 / (t - self.landmarks.flatten()))) ** 2)

    def local_solver(
        self, t_init, y, num_iters=100, eps=1e-5, W=None, verbose=False, **kwargs
    ):
        info = {}
        a = self.landmarks.flatten()
        x_op = t_init
        for i in range(num_iters):
            u = y - (1 / (x_op - a))
            if verbose:
                print(f"cost {i}", np.sum(u**2))
            du = 1 / ((x_op - a) ** 2)
            if np.linalg.norm(du) > 1e-10:
                dx = -np.sum(u * du) / np.sum(du * du)
                x_op = x_op + dx
                if np.abs(dx) < eps:
                    msg = f"converged in dx after {i} it"
                    cost = self.get_cost(x_op, y)
                    info = {"msg": msg, "cost": cost, "success": True}
                    return x_op, info, cost
            else:
                msg = f"converged in du after {i} it"
                cost = self.get_cost(x_op, y)
                info = {"msg": msg, "cost": self.get_cost(x_op, y), "success": True}
                return x_op, info, cost
        return None, {"msg": "didn't converge", "cost": None, "success": False}, None

    def __repr__(self):
        return f"stereo1d_{self.param_level}"
