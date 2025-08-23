import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from popcor.base_lifters import RangeOnlyLifter


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

    .. math:: f(\\theta) = \\sum_{n=0}^{N-1} \\sum_{k=0}^{K-1} w_{nk} || z_{nk} d_{nk} - (p_n - a_k) ||^2

    where all are as above, except:

    - :math:`\\theta` is now the flattened vector of positions :math:`p_n` and also normal vectors :math:`z_{nk}`.
    """

    TIGHTNESS = "rank"
    LEVELS = ["normals", "simple"]
    LEVEL_NAMES = {
        "normals": "$\\boldymbol{y}_n$",
        "simple": "$z_n$",
    }
    MONOMIAL_DEGREE = 1

    SCALE = 1.0

    @staticmethod
    def create_good(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_good(n_positions, n_landmarks, d)
        lifter = RangeOnlyNsqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

    @staticmethod
    def create_bad(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_bad(n_positions, n_landmarks, d)
        lifter = RangeOnlyNsqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

    @staticmethod
    def create_bad_fixed(n_positions, n_landmarks, d=2):
        landmarks, theta = RangeOnlyLifter.create_bad_fixed(n_positions, n_landmarks, d)
        lifter = RangeOnlyNsqLifter(n_positions, n_landmarks, d)
        lifter.overwrite_theta(theta)
        lifter.landmarks = landmarks
        return lifter

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
        if var_dict is None:
            var_dict = self.var_dict

        A_list = []
        if self.level == "normals":
            for i in range(self.n_positions):
                for k in range(self.n_landmarks):
                    # enforce the normal vectors are indeed unit-norm
                    A = PolyMatrix(symmetric=True)
                    A[f"z_{i}_{k}", f"z_{i}_{k}"] = np.eye(self.d)
                    A[self.HOM, self.HOM] = -1.0
                    if output_poly:
                        A_list.append(A)
                    else:
                        A_list.append(A.get_matrix(self.var_dict))

        elif self.level == "simple":
            # enfore that y_{nk}^2 = ||p_n - a_k||^2 = ||p_n||^2 - 2a_k^T p_n + ||a_k||^2
            raise NotImplementedError(
                "get_A_known not implemented yet for simple level"
            )
        return A_list, [0.0] * len(A_list)

    def get_residuals(self, t, y, ad=False):
        return super().get_residuals(t, y, ad=ad, squared=False)

    def get_cost(self, theta, y, sub_idx=None, ad=False):
        residuals = self.get_residuals(theta, y, ad=ad)
        return self.get_cost_from_res(residuals, sub_idx, ad=ad)

    def get_normals(self, theta=None):
        if theta is None:
            theta = self.theta

        if np.ndim(theta) < 2:
            theta = theta[None, :]
        # N x M x d normals.
        return (self.landmarks[None, :, :] - theta[:, None, :]) / np.linalg.norm(
            self.landmarks[None, :, :] - theta, axis=2
        )[:, :, None]

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        positions = theta.reshape(self.n_positions, -1)
        normals = self.get_normals(positions)

        x_data = []
        for val in var_subset:
            if val == "h":
                x_data += [1.0]
            elif val.startswith("x_"):
                assert theta is not None
                i = int(val.split("x_")[-1])
                x_data += list(theta[i, :])
            elif val.startswith("z_"):  # z_i_k
                i = int(val.split("_")[-2])
                k = int(val.split("_")[-1])
                z = normals[i, k, :]
                x_data += list(z)
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

    def get_Q_from_y(self, y, output_poly: bool = False):
        Q = PolyMatrix()
        for k in range(self.n_landmarks):
            for i in range(self.n_positions):
                d_ik = y[i, k]
                # || n_ik * d_ik - (m_k - x_i) || ^2
                # = d_ik^2 - 2*d_ik*n_ik' (m_k - x_i) + ||m_k - x_i||^2
                # = d_ik^2 + ||m_k||^2 - 2m_k' x_i - 2 * d_ik * m_k' n_ik + 2 * d_ik * n_ik' x_i + ||x_i||^2
                m_k = self.landmarks[k]
                b_k = d_ik**2 + np.sum(m_k**2)

                Q["h", "h"] += b_k  # d_ik^2 + ||m_k||^2
                Q["h", f"x_{i}"] += -m_k[None, :]  # -2m_k' x_i
                Q["h", f"z_{i}_{k}"] += -d_ik * m_k[None, :]  # -2 d_ik * m_k' n_ik
                Q[f"x_{i}", f"z_{i}_{k}"] += d_ik * np.eye(self.d)  # 2 d_ik * n_ik' x_i
                Q[f"x_{i}", f"x_{i}"] += np.eye(self.d)  # + ||x_i||^2

        if output_poly:
            return Q
        else:
            return Q.get_matrix(self.var_dict)

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

    def get_valid_samples(
        self,
        n_samples,
    ):
        samples = super().get_valid_samples(n_samples)

        # TODO(FD): maybe this should be moved to theta.
        normals = self.landmarks[None, :, :] - samples[:, None, :]
        normals /= np.linalg.norm(normals, axis=2)[:, :, None]
        return np.hstack([samples, normals.reshape(normals.shape[0], -1)])

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"


if __name__ == "__main__":
    lifter = RangeOnlyNsqLifter(n_positions=3, n_landmarks=4, d=2)
