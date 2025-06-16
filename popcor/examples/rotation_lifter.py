from typing import List

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from scipy.spatial.transform import Rotation as R

from popcor.base_lifters import StateLifter

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)


def get_orthogonal_constraints(key, hom, d, level):
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
    """Rotation averaging problem.

    - level "no" corresponds to the rank-1 version.
    - level "bm" corresponds to the rank-d version (bm=Bourer Monteiro, for later extension).

    """

    LEVELS = ["no", "bm"]
    HOM = "h"
    VARIABLE_LIST = [["h", "c_0"], ["h", "c_0", "c_1"]]

    # whether or not to include the determinant constraints in the known constraints.
    ADD_DETERMINANT = False

    NOISE = 1e-3

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
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
        if self.level == "no":
            var_dict = {self.HOM: 1}
            var_dict.update({f"c_{i}": self.d**2 for i in range(self.n_rot)})
        else:
            var_dict = {self.HOM: self.d}
            var_dict.update({f"c_{i}": self.d for i in range(self.n_rot)})
        return var_dict

    def sample_theta(self):
        """Generate a random new feasible point."""
        C = np.empty((self.n_rot * self.d, self.d))
        for i in range(self.n_rot):
            if self.d == 2:
                angle = np.random.uniform(0, 2 * np.pi)
                C[i * self.d : (i + 1) * self.d, :] = R.from_euler(
                    "z", angle
                ).as_matrix()[:2, :2]
            elif self.d == 3:
                C[i * self.d : (i + 1) * self.d, :] = R.random().as_matrix()
        return C

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        """Get the lifted vector x given theta and parameters."""
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
                    x_data += list(theta[i * self.d : (i + 1) * self.d].flatten("C"))
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
                    x_data.append(theta[i * self.d : (i + 1) * self.d])
                else:
                    raise ValueError(f"untreated key {key}")
            return np.vstack(x_data)
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def get_theta(self, x: np.ndarray) -> np.ndarray:
        if self.level == "no":
            if np.ndim(x) == 2:
                assert x.shape[1] == 1
            C_flat = x[1 : 1 + self.n_rot * self.d**2]
            return C_flat.reshape((self.n_rot * self.d, self.d))
        elif self.level == "bm":
            # d x d
            R0 = x[: self.d, : self.d]
            # nd x d
            Ri = np.array(x[self.d : (self.n_rot + 1) * self.d, : self.d])
            # below is a block-diagonal matrix
            return np.kron(np.eye(self.n_rot), R0.T) @ Ri
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def add_relative_measurement(self, i, j, noise) -> np.ndarray:
        """
                    _
        || R_i - R_ij @ R_j ||_F^2
                        _
        measurements: R_ij = R_i @ R_j.T
        """
        R_i = self.theta[i * self.d : (i + 1) * self.d, :]
        R_j = self.theta[j * self.d : (j + 1) * self.d, :]
        R_gt = R_i @ R_j.T

        # Generate a random small rotation as noise and apply it
        if noise > 0:
            noise_rotvec = np.random.normal(scale=noise, size=(self.d,))
            Rnoise = (
                R.from_rotvec(noise_rotvec).as_matrix()
                if self.d == 3
                else R.from_euler("z", noise_rotvec[0]).as_matrix()[:2, :2]
            )
            return R_gt @ Rnoise
        else:
            return R_gt

    def add_absolute_measurement(self, i, noise, n_meas=1) -> List[np.ndarray]:
        """_
        || R_i - R_i ||

        measurements:   _
                        R_i = R_i @ noise
        """
        R_gt = self.theta[i * self.d : (i + 1) * self.d, :]
        y = []
        for _ in range(n_meas):
            # noise model: R_i = R.T @ Rnoise
            if noise > 0:
                # Generate a random small rotation as noise and apply it
                noise_rotvec = np.random.normal(scale=noise, size=(self.d,))
                Rnoise = (
                    R.from_rotvec(noise_rotvec).as_matrix()
                    if self.d == 3
                    else R.from_euler("z", noise_rotvec[0]).as_matrix()[:2, :2]
                )
                Ri = R_gt @ Rnoise
            else:
                Ri = R_gt
            y.append(Ri)
        return y

    def simulate_y(self, noise: float | None = None) -> dict:
        if noise is None:
            noise = self.NOISE

        y = {}
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

    def get_Q(self, noise: float | None = None, output_poly: bool = False):
        if noise is None:
            noise = self.NOISE
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)

        return self.get_Q_from_y(self.y_, output_poly=output_poly)

    def get_L(self, theta=None):
        """This function is deprecated as we now introduce a homogeneous variable R0.

        Returns matrix L from the cost term, so that we can add non-quadratic terms.
        F is the fixed rotation matrix

        || R - F ||_F = tr(R'R) - 2 tr(F'R) + tr(F'F)
                      = 2 tr(I) - 2 tr(F'R)

        # R is Nd x d
        argmin "" = argmin -2 * tr(F'R_0)
                  = argmin -2 * vec(F).T @ vec(R_0)

        will add trace(L'R), so L is of shape Nd x d
        """
        if theta is None:
            theta = self.theta
        R0 = theta[: self.d, : self.d]

        if self.level == "bm":
            L = PolyMatrix(symmetric=False)
            L["c_0", "width"] = -R0
            return L.get_matrix(variables=(self.var_dict, {"width": self.d}))
        elif self.level == "no":
            L = PolyMatrix(symmetric=False)
            L["c_0", "width"] = -R0.flatten("C")
            return L.get_matrix(variables=(self.var_dict, {"width": 1}))
        else:
            raise ValueError(f"Unknown level {self.level} for RotationLifter")

    def get_Q_from_y(self, y, output_poly=False):
        """param y: list of noisy rotation matrices."""
        Q = PolyMatrix()

        for key, R in y.items():
            # treat unary factors
            # f(R) = sum_i || R  - Ri ||_F^2
            # argmin f(R) = argmin sum_i tr((R - Ri)'(R - Ri))
            #             = argmin sum_i -2 tr(Ri.T @ R)
            #     tr(A.T @ B) = vec(A).T @ vec(B)
            #             = argmin sum_i -2 vec(Ri).T @ vec(R)
            if isinstance(key, int):
                assert isinstance(R, list)
                for Ri in R:
                    if self.level == "no":
                        Q[self.HOM, f"c_{key}"] -= Ri.flatten("C")[None, :]
                        Q[self.HOM, self.HOM] += len(R) * self.d
                    elif self.level == "bm":
                        Q[self.HOM, f"c_{key}"] -= Ri
            # treat binary factors
            # f(R) = sum_ij || Ri  - Rij @ Rj ||_F^2
            # argmin f(R) = argmin sum_i -2 tr(Ri.T @ R_ij @ R_j)
            #                           = tr(R_ij @ R_j @ Ri.T)
            #     tr(A.T @ C @ B) = vec(A).T @ (I kron C) @ vec(B)
            #             = argmin sum_i -2 tr((I kron R_ij) @ vec(R_j) vec(Ri).T)
            elif isinstance(key, tuple):
                i, j = key
                if self.level == "no":
                    Q[f"c_{j}", f"c_{i}"] -= np.kron(np.eye(self.d), R)
                elif self.level == "bm":
                    Q[f"c_{j}", f"c_{i}"] -= R

        if output_poly:
            return Q
        else:
            return Q.get_matrix(self.var_dict)

    def local_solver_old(
        self, t0, y, verbose=False, method=METHOD, solver_kwargs=SOLVER_KWARGS
    ):
        """Not used anymore, kept for reference. We now use the default
        QCQP local solver."""
        import pymanopt
        from pymanopt.manifolds import SpecialOrthogonalGroup

        if method == "CG":
            from pymanopt.optimizers import ConjugateGradient as Optimizer  # fastest
        elif method == "SD":
            from pymanopt.optimizers import SteepestDescent as Optimizer  # slow
        elif method == "TR":
            from pymanopt.optimizers import TrustRegions as Optimizer  # okay
        else:
            raise ValueError(method)

        if verbose:
            solver_kwargs["verbosity"] = 2
        else:
            solver_kwargs["verbosity"] = 0

        manifold = SpecialOrthogonalGroup(self.d, k=1)

        @pymanopt.function.autograd(manifold)
        def cost(R):
            cost = 0
            for Ri in y:
                cost += np.sum((R.T @ Ri - np.eye(self.d)) ** 2)
            return cost

        euclidean_gradient = None
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        )
        optimizer = Optimizer(**solver_kwargs)

        res = optimizer.run(problem, initial_point=t0)
        theta_hat = res.point

        success = ("min step_size" in res.stopping_criterion) or (
            "min grad norm" in res.stopping_criterion
        )
        info = {
            "success": success,
            "msg": res.stopping_criterion,
        }
        if success:
            return theta_hat, info, cost

    def test_and_add(self, A_list, Ai, output_poly, b_list=[], bi=0.0):
        x = self.get_x()
        Ai_sparse = Ai.get_matrix(self.var_dict)
        err = np.trace(np.atleast_2d(x.T @ Ai_sparse @ x)) - bi
        assert abs(err) <= 1e-10, err
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)
        b_list.append(bi)

    def get_A0(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        if self.level == "no":
            return super().get_A0(var_subset=var_subset)
        else:
            # using self.HOM, None just to make it clear that the second argument is not used.
            A_orth, b_orth = get_orthogonal_constraints(
                self.HOM, None, self.d, self.level
            )
            return [Ai.get_matrix(var_subset) for Ai in A_orth], b_orth

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        A_list = []
        b_list = []
        if var_dict is None:
            var_dict = self.var_dict

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
                if self.level == "bm":
                    print("No known redundant constraints for level bm")
                    continue

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
        if self.level == "bm":
            return A_list, b_list
        else:
            return A_list

    def plot(self, estimates={}):
        import itertools

        import matplotlib.pyplot as plt

        from popcor.utils.plotting_tools import plot_frame

        fig, ax = plt.subplots()
        label = "gt"
        for i in range(self.n_rot):
            plot_frame(
                ax=ax,
                theta=self.theta[i * self.d : (i + 1) * self.d, :],
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
                    theta=theta[i * self.d : (i + 1) * self.d, :],
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
    A_known = lifter.get_A_known(output_poly=False, add_redundant=True)
    constraints = lifter.get_A_b_list(A_known)

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
