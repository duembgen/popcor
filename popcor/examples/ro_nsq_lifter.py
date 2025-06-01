import itertools

import numpy as np
import scipy.sparse as sp
from poly_matrix.least_squares_problem import LeastSquaresProblem

from popcor.base_lifters import RangeOnlyLifter
from popcor.utils.common import diag_indices

NOISE = 1e-2  # std deviation of distance noise

NORMALIZE = True


class RangeOnlyNsqLifter(RangeOnlyLifter):
    """Range-only localization in 2D or 3D.

    Almost same as RangeOnlySqLifter, but we do not square the distances. We minimize

    .. math:: f(\\theta) = \\sum_{n=0}^{N-1} \\sum_{k=0}^{K-1} w_{nk} (d_{nk} - ||p_n - a_k||)^2

    where

    - :math:`w_{nk}` is the weight for the nth point and kth landmark (currently assumed binary to mark missing edges).
    - :math:`\\theta` is the flattened vector of positions :math:`p_n`.
    - :math:`d_{nk}` is the distance measurement from point n to landmark k.
    - :math:`a_k` is the kth landmark.

    Note that in the current implementation, there is no regularization term so the problem could be split into individual points.

    We experiment with two different substitutions to turn the cost function into a quadratic form:

    - level "normals" uses a reformulation that introduce normal vectors, as proposed by Halstedt et al (see below).
    - level "simple" uses substitution :math:`z_i=||p_n - a_k||` (or equivalent 3D version).

    .. math:: f(\\theta) = \\sum_{n=0}^{N-1} \\sum_{k=0}^{K-1} w_{nk} || n_{nk} - d_{nk}\\top(p_n - a_k) ||^2

    where all are as above, except:

    - :math:`\\theta` is now the flattened vector of positions :math:`p_n` and also normal vectors :amth:`z_{nk}`.
    """

    TIGHTNESS = "rank"
    LEVELS = ["normals", "simple"]
    LEVEL_NAMES = {
        "normals": "$\\boldymbol{y}_n$",
        "simple": "$z_n$",
    }

    def __init__(
        self,
        n_positions,
        n_landmarks,
        d,
        W=None,
        level="normals",
        variable_list=None,
        param_level="no",
    ):
        if level == "simple":
            raise NotImplementedError("simple is not implemented yet.")
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
            [self.HOM, "x_0"] + [f"z_0_{i}" for i in range(self.n_landmarks)],
        ]

    def get_all_variables(self):
        vars = [self.HOM]
        vars += [f"x_{i}" for i in range(self.n_positions)]
        vars += [
            f"z_{i}_{k}"
            for i in range(self.n_positions)
            for k in range(self.n_landmarks)
        ]
        return [vars]

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        from poly_matrix.poly_matrix import PolyMatrix

        if var_dict is None:
            var_dict = self.var_dict
        positions = self.get_variable_indices(var_dict)

        A_list = []
        for n in positions:
            if self.level == "normals":
                # enforce the normal vectors are indeed unit-norm
                A = PolyMatrix(symmetric=True)

                if output_poly:
                    A_list.append(A)
                else:
                    A_list.append(A.get_matrix(self.var_dict))

            elif self.level == "simple":
                # enfore that y_{nk}^2 = ||p_n - a_k||^2 = ||p_n||^2 - 2a_k^T p_n + ||a_k||^2
                A = PolyMatrix(symmetric=True)

                if output_poly:
                    A_list.append(A)
                else:
                    A_list.append(A.get_matrix(self.var_dict))
        return A_list

    def get_residuals(self, t, y, ad=False):
        return super().get_residuals(t, y, ad=ad, squared=False)

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        positions = theta.reshape(self.n_positions, -1)
        normals = self.get_normals(positions, parameters)

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif "x" in key:
                n = int(key.split("_")[-1])
                x_data += list(positions[n])
            elif "z" in key:
                n, k = [int(k) for k in key.split("_")[-2:]]
                if self.level == "simple":
                    x_data.append(np.linalg.norm(positions[n]))
                elif self.level == "normals":
                    x_data += list((normals[n][k]))
        assert len(x_data) == self.get_dim_x(var_subset)
        return np.array(x_data)

    def get_J_lifting(self, t):
        pos = t.reshape((-1, self.d))
        ii = []
        jj = []
        data = []

        for n in range(self.n_positions):
            if self.level == "normals":
                pass
            elif self.level == "simple":
                pass
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

    def get_Q_from_y(self, y, output_poly: bool = False):
        self.ls_problem = LeastSquaresProblem()

        if self.level == "quad":
            diag_idx = diag_indices(self.d)

        for n, k in itertools.product(range(self.n_positions), range(self.n_landmarks)):
            if self.W[n, k] > 0:
                ak = self.landmarks[k]
                if self.level == "no":
                    self.ls_problem.add_residual(
                        {
                            self.HOM: y[n, k] - np.linalg.norm(ak) ** 2,
                            f"x_{n}": 2 * ak.reshape((1, -1)),
                            f"z_{n}": -1,
                        }
                    )
                elif self.level == "quad":
                    mat = np.zeros((1, self.size_z))
                    mat[0, diag_idx] = -1
                    res_dict = {
                        self.HOM: y[n, k] - np.linalg.norm(ak) ** 2,
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
        return super().simulate_y(noise=noise, squared=False)

    @property
    def var_dict(self):
        var_dict = {self.HOM: 1}
        var_dict.update({f"x_{n}": self.d for n in range(self.n_positions)})
        var_dict.update(
            {
                f"z_{n}_{k}": self.size_z
                for n in range(self.n_positions)
                for k in range(self.n_landmarks)
            }
        )
        return var_dict

    @property
    def size_z(self) -> int:
        if self.level == "normals":
            return self.d
        elif self.level == "simple":
            return 1
        else:
            raise ValueError(f"Unknown level {self.level}")

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"


if __name__ == "__main__":
    lifter = RangeOnlyNsqLifter(n_positions=3, n_landmarks=4, d=2)
