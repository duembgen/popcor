import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from scipy.spatial.transform import Rotation as R

from popcor.base_lifters import StateLifter

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)


class RotationLifter(StateLifter):
    """Rotation averaging problem."""

    LEVELS = ["no"]
    HOM = "h"
    VARIABLE_LIST = [["h", "c"]]

    # whether or not to include the determinant constraints in the known constraints.
    ADD_DETERMINANT = False

    NOISE = 1e-3

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(self, level="no", param_level="no", d=2, n_meas=2):
        self.n_meas = n_meas
        self.level = level
        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
        )

    @property
    def var_dict(self):
        return {self.HOM: 1, "c": self.d**2}

    def sample_theta(self):
        """Generate a random new feasible point."""

        if self.d == 2:
            angle = np.random.uniform(0, 2 * np.pi)
            C = R.from_euler("z", angle).as_matrix()[:2, :2]
        elif self.d == 3:
            C = R.random().as_matrix()
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
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif key == "c":
                x_data += list(theta.flatten("C"))
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_theta(self, x: np.ndarray) -> np.ndarray:
        assert np.ndim(x) == 1
        C_flat = x[1 : 1 + self.d**2]
        return C_flat.reshape((self.d, self.d))

    def simulate_y(self, noise: float | None = None):
        if noise is None:
            noise = self.NOISE

        self.y_ = []
        for i in range(self.n_meas):
            # noise model: R_i = R.T @ Rnoise
            if noise > 0:
                # Generate a random small rotation as noise and apply it
                noise_rotvec = np.random.normal(scale=noise, size=(self.d,))
                Rnoise = (
                    R.from_rotvec(noise_rotvec).as_matrix()
                    if self.d == 3
                    else R.from_euler("z", noise_rotvec[0]).as_matrix()[:2, :2]
                )
                Ri = self.theta.T @ Rnoise
            else:
                Ri = self.theta.T
            self.y_.append(Ri)
        return self.y_

    def get_Q(self, noise: float | None = None):
        if self.y_ is None:
            self.y = self.simulate_y(noise=noise)
        return self.get_Q_from_y(self.y_)

    def get_Q_from_y(self, y, output_poly=False):
        # f(R) = sum_i || R @ R_i - I ||_F^2
        # argmin f(R) = argmin sum_i || R_i.T @ R_i ||^2 - 2 tr(R.T @ R_i) + ||I||_F^2
        #             = argmin sum_i -2 tr(R.T @ R_i) + sum_i d
        #             = argmin sum_i -2 vec(R).T @ vec(R_i.T) + N * d
        # sanity check for zero noise:
        #              || R @ R.T - I ||_F^2 = 0
        """param y: list of noisy rotation matrices."""
        Q = PolyMatrix()
        for Ri in y:
            Q[self.HOM, "c"] -= Ri.T.flatten("C")[None, :]
        Q[self.HOM, self.HOM] += len(y) * self.d
        if output_poly:
            return Q
        else:
            return Q.get_matrix(self.var_dict)

    def local_solver_old(
        self, t0, y, verbose=False, method=METHOD, solver_kwargs=SOLVER_KWARGS
    ):
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

    def test_and_add(self, A_list, Ai, output_poly):
        x = self.get_x()
        Ai_sparse = Ai.get_matrix(self.var_dict)
        err = x.T @ Ai_sparse @ x
        assert abs(err) <= 1e-10, err
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        A_list = []
        if var_dict is None:
            var_dict = self.var_dict

        if "c" in var_dict:
            # enforce diagonal == 1
            for i in range(self.d):
                Ei = np.zeros((self.d, self.d))
                Ei[i, i] = 1.0
                constraint = np.kron(Ei, np.eye(self.d))
                Ai = PolyMatrix(symmetric=True)
                Ai["c", "c"] = constraint
                Ai[self.HOM, self.HOM] = -1
                self.test_and_add(A_list, Ai, output_poly=output_poly)

            # enforce off-diagonal == 0
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, j] = 1.0
                    Ei[j, i] = 1.0
                    constraint = np.kron(Ei, np.eye(self.d))
                    Ai = PolyMatrix(symmetric=True)
                    Ai["c", "c"] = constraint
                    self.test_and_add(A_list, Ai, output_poly=output_poly)

            # enforce that determinant is one.
            if self.d == 2 and self.ADD_DETERMINANT:
                # C = [a b; c d]; ad - bc - 1 = 0
                #    a b c d
                # a        1
                # b     -1
                # c   -1
                # d 1
                Ai = PolyMatrix(symmetric=True)
                constraint = np.zeros((self.d**2, self.d**2))
                constraint[0, 3] = constraint[3, 0] = 1.0
                constraint[1, 2] = constraint[2, 1] = -1.0
                Ai[self.HOM, self.HOM] = -2
                Ai["c", "c"] = constraint
                self.test_and_add(A_list, Ai, output_poly=output_poly)
            elif self.d == 3 and self.ADD_DETERMINANT:
                #      c11  c12  c13                  c21 * c32 - c31 * c22 = c13
                # C = [c21, c22, c23]; c1 x c2 = c3:  c31 * c12 - c11 * c12 = c23
                #      c31  c32  c33                  c11 * c22 - c21 * c12 = c33
                print(
                    "Warning: consider implementing the determinant constraint for RobustPoseLifter, d=3"
                )
        return A_list

    def plot(self, estimates={}):
        import matplotlib.pyplot as plt

        from popr.utils.plotting_tools import plot_frame

        fig, ax = plt.subplots()
        plot_frame(ax=ax, theta=self.theta, label="gt", ls="-", scale=0.5)

        for label, theta in estimates.items():
            plot_frame(ax=ax, theta=theta, label=label, ls="--", scale=1.0)

        ax.set_aspect("equal")
        ax.legend()
        return fig, ax
