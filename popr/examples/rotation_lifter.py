import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from scipy.spatial.transform import Rotation as R

from popr.lifters import StateLifter

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)


class RotationLifter(StateLifter):
    LEVELS = ["no"]
    PARAM_LEVELS = ["no", "p", "ppT"]

    HOM = "h"
    VARIABLE_LIST = ["h", "c"]

    # whether or not to include the determinant constraints in the known constraints.
    ADD_DETERMINANT = False

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

    def get_Q(self, noise):
        # f(R) = sum_i || R @ R_i - I ||_F^2
        # min f(R) = min || R_i.T @ R_i ||^2 - 2 tr(R.T @ R_i) + I
        #          = max tr(R.T @ R_i)
        if self.y_ is None:
            self.y_ = []
            for i in range(self.n_meas):
                if noise > 0:
                    C = self.theta.T @ R.random(noise).as_matrix()
                else:
                    C = self.theta.T
                self.y_.append(C)
        raise NotImplementedError("continue here!")

    def local_solver(
        self, t0, y, verbose=False, method=METHOD, solver_kwargs=SOLVER_KWARGS
    ):
        raise NotImplementedError("local solver not tested yet")
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
            for yi in y:
                cost += R.T @ yi - np.eye(self.d)
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
