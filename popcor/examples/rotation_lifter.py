"""
A class for solving rotation averaging problems.
"""

from typing import List

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from scipy.spatial.transform import Rotation

from popcor.base_lifters import StateLifter

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)

DEBUG = True


def get_orthogonal_constraints(key, hom, d, level):
    """Return A, b lists enforcing orthogonality constraints for variable `key`.

    Produces constraints encoding R'R = I (diagonal == 1 and off-diagonal == 0)
    for either the rank-1 ("no") or Burer-Monteiro ("bm") formulation.
    """
    A_list = []
    b_list = []
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
    by :math:`R_w^{-1}R_i` afterwards to move the world frame to the origin. Using this formulation, all
    measurements are binary factors, which may simplify implementation.

    We consider two different formulations of the problem:

    - level "no" corresponds to the rank-1 version:

    .. math::
        x = \\begin{bmatrix} 1, \\mathrm{vec}(R_1), \\ldots, \\mathrm{vec}(R_N) \\end{bmatrix}^T

    - level "bm" corresponds to the rank-d version (bm=Burer-Monteiro).

    .. math::
        X = \\begin{bmatrix} R_1^\\top \\\\ \\vdots \\\\ R_N^\\top \\end{bmatrix}

    According to the above conventions, HOM is either 1 or R_0, the world frame.
    """

    LEVELS = ["no", "bm"]
    HOM = "h"
    VARIABLE_LIST = [["h", "c_0"], ["h", "c_0", "c_1"]]

    # whether or not to include the determinant constraints in the known constraints.
    ADD_DETERMINANT = False
    NOISE = 1e-3

    def __init__(
        self,
        level="no",
        param_level="no",
        d=2,
        n_abs=2,
        n_rot=1,
        n_rel=1,
        sparsity="chain",
    ):
        """Initialize a RotationLifter instance with problem dimensions and options."""
        assert n_rel in [
            0,
            1,
        ], "do not support more than 1 relative measurement per pair currently."
        self.n_rot = n_rot
        self.n_abs = n_abs
        self.n_rel = n_rel
        self.level = level
        self.sparsity = sparsity
        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
        )

    @property
    def var_dict(self):
        """Return dictionary mapping variable names to their dimensions in the lifted representation."""
        if self.level == "no":
            # self.HOM is the scalar homogenization variable
            var_dict = {self.HOM: 1}
            var_dict.update({f"c_{i}": self.d**2 for i in range(self.n_rot)})
        else:
            # self.HOM is the world frame, which is d x d
            var_dict = {self.HOM: self.d}
            var_dict.update({f"c_{i}": self.d for i in range(self.n_rot)})
        return var_dict

    def sample_theta(self):
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

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        """Return the lifted variable x (vector or stacked matrices) for given theta and var_subset."""
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict.keys()

        x_data = []
        if self.level == "no":
            for key in var_subset:
                if key == self.HOM:
                    x_data.append(1.0)
                elif "c" in key:
                    i = int(key.split("_")[1])
                    # column-major flattening to match vec(R)
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
            # Remember that x is composed of R_0.T, R_1.T, ..., R_N.T
            # We want to return theta in the form [R*_1, R*_2, ..., R*_N]
            # where each R*_i := R_0.T @ R_i

            # d x d (original world frame)
            R0 = x[: self.d, : self.d].T

            # nd x d: [R_1.T; R_2.T; ...; R_N.T]
            Ri = np.array(x[self.d : (self.n_rot + 1) * self.d, : self.d]).T

            # return d x nd: [R*_1, R*_2, ..., R*_N]
            Ri_world = R0.T @ Ri
            return Ri_world
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def add_relative_measurement(self, i, j, noise) -> np.ndarray:
        """Create a noisy relative measurement R_ij = R_i.T @ R_j with additive rotation noise.
        error terms: || R_j - R_i @ R_ij ||_F^2
        measurements: R_ij = R_i.T @ R_j
        """
        R_i = self.theta[:, i * self.d : (i + 1) * self.d]
        R_j = self.theta[:, j * self.d : (j + 1) * self.d]
        R_gt = R_i.T @ R_j

        # Generate a random small rotation as noise and apply it
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

    def add_absolute_measurement(self, i, noise, n_meas=1) -> List[np.ndarray]:
        """Create one or more noisy absolute measurements of rotation R_i (relative to world).
        error terms: || R_i - R_w @ R_wi ||_F^2
        measurements: R_wi = R_w.T @ R_i
        """
        R_gt = self.theta[:, i * self.d : (i + 1) * self.d]
        y = []
        for _ in range(n_meas):
            # noise model: R_i = R.T @ Rnoise
            if noise > 0:
                # Generate a random small rotation as noise and apply it
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

    def simulate_y(self, noise: float | None = None) -> dict:
        """Simulate measurement dictionary y given current theta and noise level."""
        if noise is None:
            noise = self.NOISE

        y = {}
        if self.n_abs > 0:
            for i in range(self.n_rot):
                y[i] = self.add_absolute_measurement(i, noise, self.n_abs)
        else:
            # add prior if no absolute measurements exist
            y[0] = self.add_absolute_measurement(0, 0.0, 1)

        if self.n_rel > 0:
            if self.sparsity == "chain":
                for i in range(self.n_rot - 1):
                    j = i + 1
                    y[(i, j)] = self.add_relative_measurement(i, j, noise)
            else:
                raise ValueError(f"Unknown sparsity {self.sparsity}")
        return y

    def get_Q(self, noise: float | None = None, output_poly: bool = False):
        """Return the cost matrix Q (poly or ndarray) constructed from simulated measurements."""
        if noise is None:
            noise = self.NOISE
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)

        return self.get_Q_from_y(self.y_, output_poly=output_poly)

    def get_Q_from_y(self, y, output_poly=False):
        """Construct the quadratic cost (PolyMatrix or ndarray) from measurement dictionary y."""
        Q = PolyMatrix()

        for key, R in y.items():
            if isinstance(key, int):
                # loop over all absolute measurements of this rotation.
                assert isinstance(R, list)
                for Rk in R:
                    if self.level == "no":
                        # unary factors: f(R) = sum_k || R  - Rk ||_F^2 = sum_k 2tr(I) - 2tr(R'Rk)
                        # Rk is measured rotation
                        Q_test = PolyMatrix()
                        Q_test[self.HOM, f"c_{key}"] -= Rk.flatten("F")[None, :]
                        # Not adding below to be consistent with "bm" case, where we cannot
                        # add a constant term to the cost.
                        # Q_test[self.HOM, self.HOM] += 2 * self.d
                        if DEBUG:
                            x = self.get_x()
                            cost_x = x.T @ Q_test.get_matrix(self.var_dict) @ x
                            Ri = self.theta[:, key * self.d : (key + 1) * self.d]
                            cost_R = np.linalg.norm(Ri - Rk) ** 2 - 2 * self.d
                            assert abs(cost_x - cost_R) < 1e-10
                        Q += Q_test

                    elif self.level == "bm":
                        # use HOM as a world frame: f(R) = sum_i || R - HOM @ Rk ||
                        # compare with below:
                        # R corresponds to Rj, HOM corresponds to Ri
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
            # treat binary factors
            # f(R) = sum_ij || Rj  - Ri @ Rij ||_F^2
            #      = sum_ij tr((Rj - Ri @ Rij)'(Rj - Ri @ Rij))
            #      = sum_ij tr(Rj'Rj) - 2 tr(Rj'Ri Rij) + tr(Rij'Ri'RiRij)
            #      = sum_ij 2tr(I) - 2tr(Rij Rj' Ri)
            #      = sum_ij 2tr(I)  - 2c_i' (Rij kron I) c_j
            elif isinstance(key, tuple):
                i, j = key
                if self.level == "no":
                    Q_test = PolyMatrix()

                    # Not adding below to be consistent with "bm" case, where we cannot
                    # add a constant term to the cost.
                    # Q_test[self.HOM, self.HOM] += 2 * self.d
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

    def test_and_add(self, A_list, Ai, output_poly, b_list=[], bi=0.0):
        """Test that constraint Ai holds at current theta then append it to A_list and b_list."""
        x = self.get_x()
        Ai_sparse = Ai.get_matrix(self.var_dict)
        err = np.trace(np.atleast_2d(x.T @ Ai_sparse @ x)) - bi
        assert abs(err) <= 1e-10, err
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)
        b_list.append(bi)

    def get_A0(self, var_subset=None) -> tuple[list, list]:
        """Return the homogenization constraint A0 for chosen level (either h^2=1, or H'H=I)."""
        if var_subset is None:
            var_subset = self.var_dict
        if self.level == "no":
            return super().get_A0(var_subset=var_subset)
        else:
            # using self.HOM, None just to make it clear that the second argument is not used.
            A_orth, b_orth = get_orthogonal_constraints(
                self.HOM, None, self.d, self.level
            )
            return [Ai.get_matrix(var_subset) for Ai in A_orth], list(b_orth)

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        """Return known linear constraints (A and b). If level 'no' returns A_list; if 'bm' returns (A_list, b_list)."""
        A_list = []
        b_list = []
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

                # enforce that determinant is one.
                if self.d == 2 and self.ADD_DETERMINANT:
                    # level "no":
                    # C = [a b; c d]; ad - bc - 1 = 0
                    #    a b c d
                    # a        1
                    # b     -1
                    # c   -1
                    # d 1
                    # level "bm"
                    # C = [a b
                    #      c d]
                    # C @ C.T
                    # [a b] [a c]   [a^2 + b^2 a*c + b*d]
                    # [c d] [b d] = [a*c + b*d c^2 + d^2]
                    # cannot be implemented...
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
                    #      c11  c12  c13                  c21 * c32 - c31 * c22 = c13
                    # C = [c21, c22, c23]; c1 x c2 = c3:  c31 * c12 - c11 * c12 = c23
                    #      c31  c32  c33                  c11 * c22 - c21 * c12 = c33
                    print(
                        "Warning: consider implementing the determinant constraint for RobustPoseLifter, d=3"
                    )

            if add_redundant and f"c_{k}" in var_dict:
                # enforce diagonal == 1 for RR' = I
                for i in range(self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, i] = 1.0
                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, self.HOM] = -1
                    constraint = np.kron(np.eye(self.d), Ei)
                    Ai[f"c_{k}", f"c_{k}"] = constraint
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                if self.d == 2:
                    # enforce structure:
                    # [cos(x) -sin(x)]
                    # [sin(x) cos(x)]
                    # => [c s -s c]
                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, f"c_{k}"] = np.array([1.0, 0, 0, -1.0])[None, :]
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, f"c_{k}"] = np.array([0, 1.0, 1.0, 0.0])[None, :]
                    self.test_and_add(A_list, Ai, output_poly, b_list, 0.0)

                # enforce off-diagonal == 0 for RR' = I
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

    def plot(self, estimates={}):
        """Plot ground-truth frames and optional estimated frames; returns (fig, ax)."""
        import itertools

        import matplotlib.pyplot as plt

        from popcor.utils.plotting_tools import plot_frame

        fig, ax = plt.subplots()
        label = "gt"
        for i in range(self.n_rot):
            plot_frame(
                ax=ax,
                theta=self.theta[:, i * self.d : (i + 1) * self.d],
                label=label,
                ls="-",
                scale=0.5,
                marker="",
                r_wc_w=np.hstack([i * 2.0] + [0.0] * (self.d - 1)),  # type.ignore
            )
            label = None

        linestyles = itertools.cycle(["--", "-.", ":"])
        for label, theta in estimates.items():
            for i in range(self.n_rot):
                plot_frame(
                    ax=ax,
                    theta=theta[:, i * self.d : (i + 1) * self.d],
                    label=label,
                    ls=next(linestyles),
                    scale=1.0,
                    marker="",
                    r_wc_w=np.hstack([i * 2.0] + [0.0] * (self.d - 1)),  # type.ignore
                )
                label = None

        ax.set_aspect("equal")
        ax.legend()
        return fig, ax

    def __repr__(self):
        return f"rotation_lifter{self.d}d_{self.level}"


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from cert_tools.linalg_tools import rank_project
    from cert_tools.sdp_solvers import solve_sdp

    from popcor.utils.plotting_tools import plot_matrix

    level = "no"
    # level = "bm"

    np.random.seed(0)
    # lifter = RotationLifter(
    #     d=2, n_abs=0, n_rel=1, n_rot=4, sparsity="chain", level=level
    # )
    lifter = RotationLifter(
        d=2, n_abs=1, n_rel=0, n_rot=4, sparsity="chain", level=level
    )

    # add relative measurements between all subsequent rotations.
    # lifter.add_relative_measurement(0, 1, noise=1e-3)
    # add one absolute measurement to fix Gauge freedom
    # lifter.add_absolute_measurement(0, noise=1e-5)

    y = lifter.simulate_y(noise=1e-10)

    x = lifter.get_x()
    rank = x.shape[1] if np.ndim(x) == 2 else 1

    theta_gt, *_ = lifter.local_solver(lifter.theta, y, verbose=False)
    estimates = {"init gt": theta_gt}
    for i in range(0):
        theta_init = lifter.sample_theta()
        theta_i, *_ = lifter.local_solver(theta_init, y, verbose=False)
        estimates[f"init random {i}"] = theta_i

    fig, ax = lifter.plot(estimates=estimates)
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
    fig, ax = lifter.plot(estimates=estimates)

    print("done")
