import itertools

import numpy as np
import scipy.sparse as sp
from poly_matrix.least_squares_problem import LeastSquaresProblem

from popcor.base_lifters import RangeOnlyLifter
from popcor.utils.common import diag_indices, upper_triangular

NOISE = 1e-2  # std deviation of distance noise

NORMALIZE = True


class RangeOnlySqLifter(RangeOnlyLifter):
    """Range-only localization in 2D or 3D.

    We minimize the cost function

    .. math:: f(\\theta) = \\sum_{n=0}^{N-1} \\sum_{k=0}^{K-1} w_{nk} (d_{nk}^2 - ||p_n - a_k||^2)^2

    where

    - :math:`w_{nk}` is the weight for the nth point and kth landmark (currently assumed binary to mark missing edges).
    - :math:`\\theta` is the flattened vector of positions :math:`p_n`.
    - :math:`d_{nk}` is the distance measurement from point n to landmark k.
    - :math:`a_k` is the kth landmark.

    Note that in the current implementation, there is no regularization term so the problem could be split into individual points.

    We experiment with two different substitutions to turn the cost function into aquadratic form:

    - level "no" uses substitution :math:`z_i=||p_i||^2=x_i^2 + y_i^2` (or equivalent 3D version).
    - level "quad" uses substitution :math:`y_i=[x_i^2, x_iy_i, y_i^2]` (or equivalent 3D version).

    This example is treated in more details in `this paper <https://arxiv.org/abs/2308.05783>`_.
    """

    TIGHTNESS = "rank"
    LEVELS = ["no", "quad"]
    LEVEL_NAMES = {
        "no": "$z_n$",
        "quad": "$\\boldsymbol{y}_n$",
    }
    MONOMIAL_DEGREE = 2

    @staticmethod
    def create_good(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_good(n_positions, n_landmarks, d)
        lifter = RangeOnlySqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

    @staticmethod
    def create_bad(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_bad(n_positions, n_landmarks, d)
        lifter = RangeOnlySqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

    @staticmethod
    def create_bad_fixed(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_bad_fixed(n_positions, n_landmarks, d)
        lifter = RangeOnlySqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

    def __init__(
        self,
        n_positions,
        n_landmarks,
        d,
        W=None,
        level="no",
        variable_list=None,
        param_level="no",
    ):
        super().__init__(
            n_positions=n_positions,
            n_landmarks=n_landmarks,
            d=d,
            W=W,
            level=level,
            variable_list=variable_list,
            param_level=param_level,
        )

    @property
    def VARIABLE_LIST(self):
        return [
            [self.HOM, "x_0"],
            [self.HOM, "x_0", "z_0"],
        ]

    def get_all_variables(self):
        vars = [self.HOM]
        vars += [f"x_{i}" for i in range(self.n_positions)]
        vars += [f"z_{i}" for i in range(self.n_positions)]
        return [vars]

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        from poly_matrix.poly_matrix import PolyMatrix

        if var_dict is None:
            var_dict = self.var_dict
        positions = self.get_variable_indices(var_dict)

        A_list = []
        for n in positions:
            if self.level == "no":
                A = PolyMatrix(symmetric=True)
                A[f"x_{n}", f"x_{n}"] = np.eye(self.d)
                A[self.HOM, f"z_{n}"] = -0.5
                if output_poly:
                    A_list.append(A)
                else:
                    A_list.append(A.get_matrix(self.var_dict))

            elif self.level == "quad":
                count = 0
                for i in range(self.d):
                    for j in range(i, self.d):
                        A = PolyMatrix(symmetric=True)
                        mat_x = np.zeros((self.d, self.d))
                        mat_z = np.zeros((1, self.size_z))
                        if i == j:
                            mat_x[i, i] = 1.0
                        else:
                            mat_x[i, j] = 0.5
                            mat_x[j, i] = 0.5
                        mat_z[0, count] = -0.5
                        A[f"x_{n}", f"x_{n}"] = mat_x
                        A[self.HOM, f"z_{n}"] = mat_z
                        count += 1
                        if output_poly:
                            A_list.append(A)
                        else:
                            A_list.append(A.get_matrix(self.var_dict))
        return A_list

    def get_residuals(self, t, y, ad=False):
        return super().get_residuals(t, y, ad=ad, squared=True)

    def get_cost(self, theta, y, sub_idx=None, ad=False):
        residuals = self.get_residuals(theta, y, ad=ad)
        return self.get_cost_from_res(residuals, sub_idx, ad=ad)

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        positions = theta.reshape(self.n_positions, -1)

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif "x" in key:
                n = int(key.split("_")[-1])
                x_data += list(positions[n])
            elif "z" in key:
                n = int(key.split("_")[-1])
                if self.level == "no":
                    x_data.append(np.linalg.norm(positions[n]) ** 2)
                elif self.level == "quad":
                    x_data += list(upper_triangular(positions[n]))
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def get_J_lifting(self, t):
        pos = t.reshape((-1, self.d))
        ii = []
        jj = []
        data = []

        idx = 0
        for n in range(self.n_positions):
            if self.level == "no":
                ii += [n] * self.d
                jj += list(range(n * self.d, (n + 1) * self.d))
                data += list(2 * pos[n])
            elif self.level == "quad":
                # it seemed easier to do this manually that programtically
                if self.d == 3:
                    x, y, z = pos[n]
                    jj += [n * self.d + j for j in [0, 0, 1, 0, 2, 1, 1, 2, 2]]
                    data += [2 * x, y, x, z, x, 2 * y, z, y, 2 * z]
                    ii += [idx + i for i in [0, 1, 1, 2, 2, 3, 4, 4, 5]]
                elif self.d == 2:
                    x, y = pos[n]
                    jj += [n * self.d + j for j in [0, 0, 1, 1]]
                    data += [2 * x, y, x, 2 * y]
                    ii += [idx + i for i in [0, 1, 1, 2]]
                idx += self.size_z
        J_lifting = sp.csr_array(
            (data, (ii, jj)),
            shape=(self.M, self.N),
        )
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for n in range(self.n_positions):
            idx = range(n * self.d, (n + 1) * self.d)
            if self.level == "no":
                hessian = sp.csr_array(
                    ([2] * self.d, (idx, idx)),
                    shape=(self.N, self.N),
                )
                hessians.append(hessian)
            elif self.level == "quad":
                for h in self.fixed_hessian_list:
                    ii, jj = np.meshgrid(idx, idx)
                    hessian = sp.csr_array(
                        (h.flatten(), (ii.flatten(), jj.flatten())),
                        shape=(self.N, self.N),
                    )
                    hessians.append(hessian)
        return hessians

    @property
    def fixed_hessian_list(self):
        if self.d == 2:
            return [
                np.array([[2, 0], [0, 0]]),
                np.array([[0, 1], [1, 0]]),
                np.array([[0, 0], [0, 2]]),
            ]
        elif self.d == 3:
            return [
                np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
                np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2]]),
            ]
        else:
            raise ValueError(f"Unsupported dimension {self.d} for fixed hessians.")

    def get_Q_from_y(self, y, output_poly: bool = False):
        """
        :param y: the distance measurements, shape (n_positions, n_landmarks). IMPORTANT: these are not squared!
        """
        self.ls_problem = LeastSquaresProblem()

        if self.level == "quad":
            diag_idx = diag_indices(self.d)

        for n, k in itertools.product(range(self.n_positions), range(self.n_landmarks)):
            if self.W[n, k] > 0:
                ak = self.landmarks[k]
                if self.level == "no":
                    self.ls_problem.add_residual(
                        {
                            self.HOM: y[n, k] ** 2 - np.linalg.norm(ak) ** 2,
                            f"x_{n}": 2 * ak.reshape((1, -1)),
                            f"z_{n}": -1,
                        }
                    )
                elif self.level == "quad":
                    mat = np.zeros((1, self.size_z))
                    mat[0, diag_idx] = -1
                    res_dict = {
                        self.HOM: y[n, k] ** 2 - np.linalg.norm(ak) ** 2,
                        f"x_{n}": 2 * ak.reshape((1, -1)),
                        f"z_{n}": mat,
                    }
                    self.ls_problem.add_residual(res_dict)
        if output_poly:
            Q = self.ls_problem.get_Q()
        else:
            Q = self.ls_problem.get_Q().get_matrix(self.var_dict)
        if NORMALIZE:
            return Q / np.sum(self.W > 0)
        return Q

    def simulate_y(self, noise: float | None = None, squared: bool = True):
        # purposefully not using squared=True here:
        # the noise should always be added to the non-squared distances.
        return super().simulate_y(noise=noise, squared=False)

    @property
    def var_dict(self):
        var_dict = {self.HOM: 1}
        var_dict.update({f"x_{n}": self.d for n in range(self.n_positions)})
        var_dict.update({f"z_{n}": self.size_z for n in range(self.n_positions)})
        return var_dict

    @property
    def size_z(self) -> int:
        if self.level == "no":
            return 1
        elif self.level == "quad":
            return int(self.d * (self.d + 1) / 2)
        else:
            raise ValueError(f"Unknown level {self.level}")

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"

    # ============ below are currently not used anymore, but it is an elegant way to compute the  =============
    #                        gradient and hessian when there are no constraints
    def get_grad(self, t, y, sub_idx=None):
        J = self.get_J(t, y)
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        if sub_idx is None:
            return 2 * J.T @ Q @ x
        else:
            sub_idx_x = self.get_sub_idx_x(sub_idx)
            return 2 * J.T[:, sub_idx_x] @ Q[sub_idx_x, :][:, sub_idx_x] @ x[sub_idx_x]  # type: ignore

    def get_J(self, t, y):
        J = sp.csr_array(
            (np.ones(self.N), (range(1, self.N + 1), range(self.N))),
            shape=(self.N + 1, self.N),
        )
        J_lift = self.get_J_lifting(t)
        J = sp.vstack([J, J_lift])
        return J

    def get_hess(self, t, y):
        x = self.get_x(t)
        Q = self.get_Q_from_y(y)
        J = self.get_J(t, y)
        hess = 2 * J.T @ Q @ J

        hessians = self.get_hess_lifting(t)
        B = self.ls_problem.get_B_matrix(self.var_dict)
        residuals = B @ x
        for m, h in enumerate(hessians):
            bm_tilde = B[:, -self.M + m]
            factor = float(bm_tilde.T @ residuals)
            hess += 2 * factor * h
        return hess

    def get_D(self, that):
        D = np.eye(1 + self.n_positions * self.d + self.size_z)
        x = self.get_x(theta=that)
        J = self.get_J_lifting(t=that)

        D = sp.lil_array((len(x), len(x)))
        D[range(len(x)), range(len(x))] = 1.0
        D[:, 0] = x
        D[-J.shape[0] :, 1 : 1 + J.shape[1]] = J  # type: ignore
        return D.tocsc()


if __name__ == "__main__":
    lifter = RangeOnlySqLifter(n_positions=3, n_landmarks=4, d=2)
