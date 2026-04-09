"""RotationAveraging class for rotation averaging and synchronization problems."""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from popcor.base_lifters import StateLifter
from popcor.utils.plotting_tools import LINESTYLES

METHOD: str = "CG"
SOLVER_KWARGS: dict = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)

DEBUG: bool = True


def get_orthogonal_constraints(
    key: Any, hom: Any, d: int, level: str
) -> Tuple[List[PolyMatrix], List[float]]:
    """Return A, b lists enforcing orthogonality constraints for variable `key`.

    Produces constraints encoding R'R = I (diagonal == 1 and off-diagonal == 0)
    for either the rank-1 ("no") or Burer-Monteiro ("bm") formulation.
    """
    A_list: List[PolyMatrix] = []
    b_list: List[float] = []
    for i in range(d):
        # enforce diagonal == 1 for R'R = I
        Ei = np.zeros((d, d))
        Ei[i, i] = 1.0
        Ai = PolyMatrix(symmetric=True)
        if level == "no":
            constraint = np.kron(Ei, np.eye(d))
            Ai[hom, hom] = -1
            Ai[key, key] = constraint
            A_list.append(Ai)
            b_list.append(0.0)
        else:
            Ai[key, key] = Ei
            A_list.append(Ai)
            b_list.append(1.0)

    # enforce off-diagonal == 0 for R'R = I
    for i in range(d):
        for j in range(i + 1, d):
            Ei = np.zeros((d, d))
            Ei[i, j] = 1.0
            Ei[j, i] = 1.0
            Ai = PolyMatrix(symmetric=True)
            if level == "no":
                constraint = np.kron(Ei, np.eye(d))
                Ai[key, key] = constraint
                A_list.append(Ai)
                b_list.append(0.0)
            else:
                Ai[key, key] = Ei
                A_list.append(Ai)
                b_list.append(0.0)
    return A_list, b_list


class RotationLifter(StateLifter):
    """Rotation averaging problem lifter.

    We solve the following optimization problem:

    .. math::
        f(\\theta) = \\min_{R_0, R_1, \\ldots, R_N \\in \\mathrm{SO}^d}
        \\sum_{i,j \\in \\mathcal{E}} || R_i - R_j \\tilde{R}_{ij} ||_F^2
        + \\sum_{i=\\in\\mathcal{A}} || R_i - \\tilde{R}_i ||_F^2

    where :math:`\\tilde{R}_{ij}` are the relative measurements, :math:`\\tilde{R}_{i}` are the absolute measurements,
    and the unknowns are

    .. math::
        \\theta = \\begin{bmatrix} R_1 & R_2 & \\ldots & R_N \\end{bmatrix}


    We can alternatively replace the absolute-measurement terms by

    .. math::
        || R_i - R_w \\tilde{R}_{i} ||_F^2

    where :math:`R_w` is an arbitrary world frame that we can also optimize over, transforming the solutions
    by :math:`R_w^{-1}R_i` after to move the world frame to the origin. Using this formulation, all
    measurements are binary factors, which may simplify implementation.

    We consider two different formulations of the problem:

    - level "no" corresponds to the rank-1 version:

    .. math::
        x = \\begin{bmatrix} 1, \\mathrm{vec}(R_1), \\ldots, \\mathrm{vec}(R_N) \\end{bmatrix}^T

    - level "bm" corresponds to the rank-d version (bm=Burer-Monteiro).

    .. math::
        X = \\begin{bmatrix} R_1^\\top \\\\ \\vdots \\\\ R_N^\\top \\end{bmatrix}
    """

    LEVELS: List[str] = ["no", "bm"]
    HOM: str = "h"
    VARIABLE_LIST: List[List[str]] = [["h", "c_0"], ["h", "c_0", "c_1"]]
    EXAMPLE_TYPES: Tuple[str, str] = ("A", "B")
    example_type: str | None = None

    ADD_DETERMINANT: bool = False
    NOISE: float = 1e-3

    def __init__(
        self,
        level: str = "no",
        param_level: str = "no",
        d: int = 2,
        n_abs: int = 2,
        n_rot: int = 1,
        n_rel: int = 1,
        sparsity: str = "chain",
    ) -> None:
        assert n_rel in [
            0,
            1,
        ], "do not support more than 1 relative measurement per pair currently."
        self.n_rot: int = n_rot
        self.n_abs: int = n_abs
        self.n_rel: int = n_rel
        self.level: str = level
        self.sparsity: str = sparsity
        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
        )

    @staticmethod
    def create_example(example_type: str = "A") -> "RotationLifter":
        """Create a deterministic SO(2) tutorial example of the requested type."""
        if example_type not in RotationLifter.EXAMPLE_TYPES:
            raise ValueError(
                f"Unknown example_type {example_type}. Expected one of {RotationLifter.EXAMPLE_TYPES}."
            )

        lifter = RotationLifter(d=2, n_rot=1, n_abs=0, n_rel=0, level="no")
        lifter.example_type = example_type

        # For tutorial examples, pin theta to the true minimizer of the SO(2)
        # objective itself.
        theta_star, _ = lifter.get_so2_global_minimum()
        lifter.theta_ = lifter.so2_theta(theta_star)
        return lifter

    @property
    def var_dict(self) -> Dict[str, int]:
        """Return dictionary mapping variable names to their dimensions in the lifted representation."""
        if self.level == "no":
            var_dict = {self.HOM: 1}
            var_dict.update({f"c_{i}": int(self.d**2) for i in range(self.n_rot)})
        else:
            var_dict = {self.HOM: int(self.d)}
            var_dict.update({f"c_{i}": int(self.d) for i in range(self.n_rot)})
        return var_dict

    def sample_theta(self) -> np.ndarray:
        """Generate a random feasible set of rotations theta shaped (d, n_rot * d)."""
        C = np.empty((self.d, self.n_rot * self.d))
        for i in range(self.n_rot):
            if self.d == 2:
                angle = np.random.uniform(0, 2 * np.pi)
                C[:, i * self.d : (i + 1) * self.d] = Rotation.from_euler(
                    "z", angle
                ).as_matrix()[:2, :2]
            elif self.d == 3:
                C[:, i * self.d : (i + 1) * self.d] = Rotation.random().as_matrix()
        return C

    @staticmethod
    def so2_theta(angle: float) -> np.ndarray:
        """Return a 2x2 SO(2) rotation matrix for a scalar angle in radians."""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def theta_so2(
        c: float,
        s: float,
        cs_block: np.ndarray | None = None,
        eps: float = 1e-8,
    ) -> float:
        """Return SO(2) angle from cosine/sine moments.

        If (c, s) is near zero and `cs_block` is provided, uses the dominant
        eigenvector of the 2x2 block as a stable fallback.
        """
        c_val = float(np.squeeze(c))
        s_val = float(np.squeeze(s))

        if np.hypot(c_val, s_val) < eps and cs_block is not None:
            block = np.asarray(cs_block, dtype=float)
            if block.shape != (2, 2):
                raise ValueError(f"cs_block must have shape (2, 2), got {block.shape}.")
            evals, evecs = np.linalg.eigh(block)
            c_val, s_val = evecs[:, int(np.argmax(evals))]

        if np.hypot(c_val, s_val) < eps:
            warnings.warn(
                "(c, s) is near zero in theta_so2; returning 0.0 rad.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0

        return float(np.arctan2(s_val, c_val))

    def get_so2_global_minimum(self, n_init: int = 2048) -> tuple[float, float]:
        """Return (theta_star, cost_star) by minimizing the SO(2) objective over angle."""
        if not (self.d == 2 and self.n_rot == 1 and self.level == "no"):
            raise ValueError(
                "SO(2) global minimum search is implemented only for d=2, n_rot=1, level='no'."
            )

        def so2_cost(angle: float) -> float:
            return float(self.get_cost(self.so2_theta(angle)))

        n = max(64, int(n_init))
        thetas = np.linspace(-np.pi, np.pi, n, endpoint=False)
        costs = np.array([so2_cost(angle) for angle in thetas])

        best_theta = float(thetas[int(np.argmin(costs))])
        best_cost = float(np.min(costs))
        delta = 2.0 * np.pi / n

        # Refine from several strong candidates to avoid local-trap issues.
        candidate_ids = np.argsort(costs)[: min(16, n)]
        for idx in candidate_ids:
            theta0 = float(thetas[int(idx)])
            left = theta0 - delta
            right = theta0 + delta
            res = minimize_scalar(so2_cost, bounds=(left, right), method="bounded")
            if res.success and float(res.fun) < best_cost:
                best_cost = float(res.fun)
                best_theta = float(res.x)

        # Wrap angle to [-pi, pi) for stable reporting.
        best_theta = float((best_theta + np.pi) % (2.0 * np.pi) - np.pi)
        return best_theta, best_cost

    def get_x(
        self,
        theta: np.ndarray | None = None,
        parameters: Any = None,
        var_subset: Any = None,
    ) -> np.ndarray:
        """Return the lifted variable x (vector or stacked matrices) for given theta and var_subset."""
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict.keys()

        x_data: list = []
        if self.level == "no":
            for key in var_subset:
                if key == self.HOM:
                    x_data.append(1.0)
                elif "c" in key:
                    i = int(key.split("_")[1])
                    x_data += list(theta[:, i * self.d : (i + 1) * self.d].flatten("F"))
                else:
                    raise ValueError(f"untreated key {key}")
            dim_x = self.get_dim_x(var_subset=var_subset)
            assert len(x_data) == dim_x
            return np.hstack(x_data)
        elif self.level == "bm":
            for key in var_subset:
                if key == self.HOM:
                    x_data.append(np.eye(self.d))
                elif "c" in key:
                    i = int(key.split("_")[1])
                    x_data.append(theta[:, i * self.d : (i + 1) * self.d].T)
                else:
                    raise ValueError(f"untreated key {key}")
            return np.vstack(x_data)
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def get_theta(self, x: np.ndarray) -> np.ndarray:
        """Recover theta (d x n_rot*d) from lifted variable x for current level."""
        if self.level == "no":
            if np.ndim(x) == 2:
                assert x.shape[1] == 1
                x = x.flatten()
            C_flat = x[1 : 1 + self.n_rot * self.d**2]
            return C_flat.reshape((self.d, self.n_rot * self.d), order="F")
        elif self.level == "bm":
            R0 = x[: self.d, : self.d].T

            I = R0.T @ R0
            scale_R = I[0, 0]
            np.testing.assert_allclose(np.diag(I), scale_R)
            warnings.warn(f"R0 is scaled by {scale_R}", UserWarning)

            Ri = np.array(x[self.d : (self.n_rot + 1) * self.d, : self.d]).T
            np.testing.assert_allclose(np.diag(Ri.T @ Ri), scale_R)
            Ri_world = R0.T @ Ri / scale_R
            np.testing.assert_allclose(np.diag(Ri_world.T @ Ri_world), 1.0)
            return Ri_world
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def add_relative_measurement(self, i: int, j: int, noise: float) -> np.ndarray:
        """Create a noisy relative measurement R_ij = R_i.T @ R_j with additive rotation noise."""
        R_i = self.theta[:, i * self.d : (i + 1) * self.d]
        R_j = self.theta[:, j * self.d : (j + 1) * self.d]
        R_gt = R_i.T @ R_j

        if noise > 0:
            noise_rotvec = np.random.normal(scale=noise, size=(self.d,))
            Rnoise = (
                Rotation.from_rotvec(noise_rotvec).as_matrix()
                if self.d == 3
                else Rotation.from_euler("z", noise_rotvec[0]).as_matrix()[:2, :2]
            )
            return R_gt @ Rnoise
        else:
            return R_gt

    def add_absolute_measurement(
        self, i: int, noise: float, n_meas: int = 1
    ) -> List[np.ndarray]:
        """Create one or more noisy absolute measurements of rotation R_i (relative to world)."""
        R_gt = self.theta[:, i * self.d : (i + 1) * self.d]
        y: List[np.ndarray] = []
        for _ in range(n_meas):
            if noise > 0:
                noise_rotvec = np.random.normal(scale=noise, size=(self.d,))
                Rnoise = (
                    Rotation.from_rotvec(noise_rotvec).as_matrix()
                    if self.d == 3
                    else Rotation.from_euler("z", noise_rotvec[0]).as_matrix()[:2, :2]
                )
                Ri = R_gt @ Rnoise
            else:
                Ri = R_gt
            y.append(Ri)
        return y

    def simulate_y(self, noise: float | None = None) -> Dict[Any, Any]:
        """Simulate measurement dictionary y given current theta and noise level."""
        if noise is None:
            noise = self.NOISE

        y: Dict[Any, Any] = {}
        if self.n_abs > 0:
            for i in range(self.n_rot):
                y[i] = self.add_absolute_measurement(i, noise, self.n_abs)
        else:
            y[0] = self.add_absolute_measurement(0, 0.0, 1)

        if self.n_rel > 0:
            if self.sparsity == "chain":
                for i in range(self.n_rot - 1):
                    j = i + 1
                    y[(i, j)] = self.add_relative_measurement(i, j, noise)
            else:
                raise ValueError(f"Unknown sparsity {self.sparsity}")
        return y

    def get_Q(
        self, noise: float | None = None, output_poly: bool = False
    ) -> PolyMatrix | np.ndarray | sp.csr_matrix | sp.csc_matrix:
        """Return the cost matrix Q (poly or ndarray) constructed from simulated measurements."""
        if getattr(self, "example_type", None) is not None:
            if not (self.d == 2 and self.n_rot == 1 and self.level == "no"):
                raise ValueError(
                    "SO(2) example Q is currently implemented only for d=2, n_rot=1, level='no'."
                )

            if self.example_type == "A":
                q2 = np.array([[0.8, 0.05], [0.05, 0.2]])
                q1 = np.array([0.0, 0.0])
            elif self.example_type == "B":
                q2 = np.array([[0.8, 0.05], [0.05, 0.2]])
                q1 = np.array([0.6, 0.0])
            else:
                raise ValueError(
                    f"Unknown example_type {self.example_type}. Expected one of {self.EXAMPLE_TYPES}."
                )

            Q = PolyMatrix(symmetric=True)

            # Vectorization is column-major: [R11, R21, R12, R22].
            q2_full = np.zeros((self.d**2, self.d**2))
            q2_full[:2, :2] = q2
            Q["c_0", "c_0"] = q2_full

            # Linear terms are represented through the homogeneous block.
            q1_full = np.zeros((1, self.d**2))
            q1_full[0, :2] = 0.5 * q1
            Q[self.HOM, "c_0"] = q1_full

            if output_poly:
                return Q
            return Q.get_matrix(self.var_dict)

        if noise is None:
            noise = self.NOISE
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)

        return self.get_Q_from_y(self.y_, output_poly=output_poly)

    def get_Q_from_y(
        self, y: Dict[Any, Any], output_poly: bool = False
    ) -> PolyMatrix | np.ndarray | sp.csr_matrix | sp.csc_matrix:
        """Construct the quadratic cost (PolyMatrix or ndarray) from measurement dictionary y."""
        Q = PolyMatrix()

        for key, R in y.items():
            if isinstance(key, int):
                assert isinstance(R, list)
                for Rk in R:
                    if self.level == "no":
                        Q_test = PolyMatrix()
                        Q_test[self.HOM, f"c_{key}"] -= Rk.flatten("F")[None, :]
                        if DEBUG:
                            x = self.get_x()
                            cost_x = x.T @ Q_test.get_matrix(self.var_dict) @ x
                            Ri = self.theta[:, key * self.d : (key + 1) * self.d]
                            cost_R = np.linalg.norm(Ri - Rk) ** 2 - 2 * self.d
                            assert abs(cost_x - cost_R) < 1e-10
                        Q += Q_test

                    elif self.level == "bm":
                        Q_test = PolyMatrix()
                        Q_test[self.HOM, f"c_{key}"] -= Rk
                        if DEBUG:
                            x = self.get_x()
                            cost_x = np.trace(
                                x.T @ Q_test.get_matrix(self.var_dict) @ x
                            )
                            Ri = self.theta[:, key * self.d : (key + 1) * self.d]
                            cost_R = np.linalg.norm(Ri - Rk) ** 2 - 2 * self.d
                            assert abs(cost_x - cost_R) < 1e-10
                        Q += Q_test
            elif isinstance(key, tuple):
                i, j = key
                if self.level == "no":
                    Q_test = PolyMatrix()
                    Q_test[f"c_{i}", f"c_{j}"] = -np.kron(R, np.eye(self.d))
                    if DEBUG:
                        x = self.get_x()
                        assert (
                            x[1:].T @ x[1:] - np.trace(self.theta.T @ self.theta)
                        ) < 1e-10
                        Ri = self.theta[:, i * self.d : (i + 1) * self.d]
                        Rj = self.theta[:, j * self.d : (j + 1) * self.d]
                        c_j = x[1 + j * self.d**2 : 1 + (j + 1) * self.d**2]
                        c_i = x[1 + i * self.d**2 : 1 + (i + 1) * self.d**2]
                        np.testing.assert_allclose(Ri.flatten("F"), c_i, atol=1e-5)
                        np.testing.assert_allclose(Rj.flatten("F"), c_j, atol=1e-5)

                        tr_R = np.trace(R @ Rj.T @ Ri)
                        tr_c = c_i.T @ (np.kron(R, np.eye(self.d)) @ c_j)
                        assert abs(tr_R - tr_c) < 1e-10

                        cost_x = x.T @ Q_test.get_matrix(self.var_dict) @ x
                        cost_R = np.linalg.norm(Rj - Ri @ R) ** 2 - 2 * self.d
                        assert abs(cost_x - cost_R) < 1e-10
                    Q += Q_test
                elif self.level == "bm":
                    Q_test = PolyMatrix()
                    Q_test[f"c_{i}", f"c_{j}"] = -R
                    if DEBUG:
                        x = self.get_x()
                        Ri = self.theta[:, i * self.d : (i + 1) * self.d]
                        Rj = self.theta[:, j * self.d : (j + 1) * self.d]

                        cost_x = np.trace(x.T @ Q_test.get_matrix(self.var_dict) @ x)
                        cost_R = np.linalg.norm(Rj - Ri @ R) ** 2 - 2 * self.d
                        assert abs(cost_x - cost_R) < 1e-9
                    Q += Q_test

        if output_poly:
            return Q
        else:
            return Q.get_matrix(self.var_dict)

    def test_and_add(
        self,
        A_list: list,
        Ai: PolyMatrix,
        output_poly: bool,
        b_list: list = [],
        bi: float = 0.0,
    ) -> None:
        x = self.get_x()
        Ai_sparse = Ai.get_matrix(self.var_dict)
        err = np.trace(np.atleast_2d(x.T @ Ai_sparse @ x)) - bi
        assert abs(err) <= 1e-10, err
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)
        b_list.append(bi)

    def get_A0(self, var_subset: Any = None) -> Tuple[list, list]:
        """Return the homogenization constraint A0 for chosen level (either h^2=1, or H'H=I)."""
        if var_subset is None:
            var_subset = self.var_dict
        if self.level == "no":
            return super().get_A0(var_subset=var_subset)
        else:
            A_orth, b_orth = get_orthogonal_constraints(
                self.HOM, None, self.d, self.level
            )
            return [Ai.get_matrix(var_subset) for Ai in A_orth], list(b_orth)

    def get_A_known(
        self,
        var_dict: Dict[str, int] | None = None,
        output_poly: bool = False,
        add_redundant: bool = False,
    ) -> Tuple[list, list]:
        """Return known linear constraints (A and b). If level 'no' returns A_list; if 'bm' returns (A_list, b_list)."""
        A_list: list = []
        b_list: list = []
        if var_dict is None:
            var_dict = self.var_dict

        if self.level == "bm" and add_redundant:
            print("No known redundant constraints for level bm")
            add_redundant = False

        for k in range(self.n_rot):
            if f"c_{k}" in var_dict:
                A_orth, b_orth = get_orthogonal_constraints(
                    f"c_{k}", self.HOM, self.d, self.level
                )
                for Ai, bi in zip(A_orth, b_orth):
                    self.test_and_add(A_list, Ai, output_poly, b_list, bi)

                if self.d == 2 and self.ADD_DETERMINANT:
                    if self.level == "bm":
                        raise NotImplementedError(
                            "Cannot add determinant constraint for level bm"
                        )
                    Ai = PolyMatrix(symmetric=True)
                    constraint = np.zeros((self.d**2, self.d**2))
                    constraint[0, 3] = constraint[3, 0] = 1.0
                    constraint[1, 2] = constraint[2, 1] = -1.0
                    Ai[self.HOM, self.HOM] = -2
                    Ai[f"c_{k}", f"c_{k}"] = constraint
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)
                elif self.d == 3 and self.ADD_DETERMINANT:
                    print(
                        "Warning: consider implementing the determinant constraint for RobustPoseLifter, d=3"
                    )

            if add_redundant and f"c_{k}" in var_dict:
                for i in range(self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, i] = 1.0
                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, self.HOM] = -1
                    constraint = np.kron(np.eye(self.d), Ei)
                    Ai[f"c_{k}", f"c_{k}"] = constraint
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                if self.d == 2:
                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, f"c_{k}"] = np.array([1.0, 0, 0, -1.0])[None, :]
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, f"c_{k}"] = np.array([0, 1.0, 1.0, 0.0])[None, :]
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                for i in range(self.d):
                    for j in range(i + 1, self.d):
                        Ei = np.zeros((self.d, self.d))
                        Ei[i, j] = 1.0
                        Ei[j, i] = 1.0
                        constraint = np.kron(np.eye(self.d), Ei)
                        Ai = PolyMatrix(symmetric=True)
                        Ai[f"c_{k}", f"c_{k}"] = constraint
                        self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)
        return A_list, b_list

    def plot_cost(
        self,
        thetas: np.ndarray | None = None,
        label: str | None = None,
        y: np.ndarray | None = None,
        xlims: tuple[float, float] | None = None,
        ylims: tuple[float, float] | None = None,
        grid_size: int | None = None,
        polar: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """Plot the cost profile for d=2 single-rotation examples."""
        import warnings

        import matplotlib.pyplot as plt

        if not (self.d == 2 and self.n_rot == 1 and self.level == "no"):
            raise ValueError(
                "Cost plotting is implemented only for d=2, n_rot=1, level='no'."
            )
        if y is not None:
            warnings.warn(
                "y is ignored in plot_cost for RotationLifter.",
                RuntimeWarning,
                stacklevel=2,
            )

        n_points = grid_size if grid_size is not None else 500
        if thetas is None:
            thetas = np.linspace(-np.pi, np.pi, n_points)
        elif grid_size is not None:
            warnings.warn(
                "grid_size is ignored when thetas is provided in plot_cost.",
                RuntimeWarning,
                stacklevel=2,
            )

        costs = np.array([self.get_cost(self.so2_theta(angle)) for angle in thetas])

        if polar:
            if xlims is not None:
                warnings.warn(
                    "xlims is ignored when polar=True in plot_cost.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            if ylims is not None:
                ax.set_ylim(*ylims)
        else:
            fig, ax = plt.subplots()
            if xlims is not None:
                ax.set_xlim(*xlims)
            if ylims is not None:
                ax.set_ylim(*ylims)
            ax.set_xlabel("theta (rad)")
            ax.set_ylabel("cost")

        ax.plot(thetas, costs, label=label)
        if label is not None:
            ax.legend()
        line = ax.lines[-1] if len(ax.lines) else None
        return fig, ax, line

    def plot_setup(
        self, estimates: Dict[str, np.ndarray] | None = None
    ) -> Tuple[Any, Any]:
        """Plot ground-truth frames and optional estimated frames."""
        import itertools

        import matplotlib.pyplot as plt

        from popcor.utils.plotting_tools import plot_frame

        if estimates is None:
            estimates = {}

        fig, ax = plt.subplots()
        gt_label = "gt"
        for i in range(self.n_rot):
            plot_frame(
                ax=ax,
                theta=self.theta[:, i * self.d : (i + 1) * self.d],
                label=gt_label,
                ls="-",
                scale=0.5,
                marker="",
                r_wc_w=np.hstack([i * 2.0] + [0.0] * (self.d - 1)),  # type: ignore
            )
            gt_label = None

        linestyles = itertools.cycle(LINESTYLES)
        for label, theta in estimates.items():
            ls = next(linestyles)
            for i in range(self.n_rot):
                plot_frame(
                    ax=ax,
                    theta=theta[:, i * self.d : (i + 1) * self.d],
                    label=label,
                    ls=ls,
                    scale=1.0,
                    marker="",
                    r_wc_w=np.hstack([i * 2.0] + [0.0] * (self.d - 1)),  # type: ignore
                )
                label = None

        ax.set_aspect("equal")
        ax.legend()
        return fig, ax

    def plot(
        self,
        thetas: np.ndarray | None = None,
        label: str | None = None,
        estimates: Dict[str, np.ndarray] | None = None,
    ) -> Tuple[Any, Any]:
        """Compatibility wrapper for plot_cost/plot_setup."""
        if thetas is not None:
            return self.plot_cost(thetas=thetas, label=label)
        return self.plot_setup(estimates=estimates)

    def __repr__(self) -> str:
        return f"rotation_lifter{self.d}d_{self.level}"


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    from cert_tools.linalg_tools import rank_project
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.utils.plotting_tools import plot_matrix

    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    angles: np.ndarray = np.linspace(-np.pi, np.pi, 500)
    for example_type in RotationLifter.EXAMPLE_TYPES:
        lifter = RotationLifter.create_example(example_type=example_type)
        fig, ax, line = lifter.plot_cost(angles, label=f"example {example_type}")
        ax.set_title(f"SO(2) example {example_type}")
        fig.savefig(
            os.path.join(
                base_dir,
                "docs",
                "source",
                "_static",
                f"rotation_lifter_{example_type}.png",
            )
        )

    # Create estimation pipeline
    level: str = "no"
    np.random.seed(0)
    lifter = RotationLifter(
        d=2, n_abs=1, n_rel=0, n_rot=4, sparsity="chain", level=level
    )

    y = lifter.simulate_y(noise=1e-10)

    x = lifter.get_x()
    rank = x.shape[1] if np.ndim(x) == 2 else 1

    theta_gt, *_ = lifter.local_solver(lifter.theta, y, verbose=False)
    estimates = {"init gt": theta_gt}
    for i in range(0):
        theta_init = lifter.sample_theta()
        theta_i, *_ = lifter.local_solver(theta_init, y, verbose=False)
        estimates[f"init random {i}"] = theta_i

    fig, ax = lifter.plot_setup(estimates=estimates)
    ax.legend()
    plt.show(block=False)

    Q = lifter.get_Q_from_y(y=y, output_poly=False)
    A_known, b_known = lifter.get_A_known(output_poly=False, add_redundant=True)
    A_0, b_0 = lifter.get_A0()
    constraints = list(zip(A_known + A_0, b_known + b_0))

    fig, axs = plt.subplots(1, len(constraints) + 1)
    fig.set_size_inches(3 * (len(constraints) + 1), 3)
    for i in range(len(constraints)):
        Ai = constraints[i][0]
        plot_matrix(Ai.toarray(), ax=axs[i], title=f"A{i} ", colorbar=False)  # type: ignore

    fig = plot_matrix(Q.toarray(), ax=axs[i + 1], title="Q", colorbar=False)  # type: ignore
    plt.show(block=False)

    X, info = solve_sdp(Q, constraints, verbose=False)

    x, info_rank = rank_project(X, p=rank)
    print(f"EVR: {info_rank['EVR']:.2e}")

    if rank == 1:
        theta_opt = lifter.get_theta(x.flatten())
    else:
        theta_opt = lifter.get_theta(x[:, :rank])

    estimates = {"init gt": theta_gt, "SDP": theta_opt}
    fig, ax = lifter.plot_setup(estimates=estimates)

    print("done")
