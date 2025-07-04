from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from scipy.spatial.transform import Rotation as R

from popcor.utils.geometry import (
    get_C_r_from_theta,
    get_noisy_pose,
    get_pose_errors_from_theta,
    get_theta_from_C_r,
)

from .state_lifter import StateLifter

N_TRYS = 10

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)

# TODO(FD) we need to add a penalty here, otherwise the local solution is not good.
# However, the penalty results in inequality constraints etc. and that's not easy to deal with.
PENALTY_RHO = 10
PENALTY_U = 1e-3

# the cutoff parameter of least squares. If residuals are >= BETA, they are considered outliers.
BETA = 0.1


class RobustPoseLifter(StateLifter, ABC):
    LEVELS = ["no", "xwT", "xxT"]
    PARAM_LEVELS = ["no", "p", "ppT"]
    LEVEL_NAMES = {"no": "no", "xwT": "x kron w", "xxT": "x kron x"}
    MAX_DIST = 10.0  # maximum of norm of t.

    @property
    def VARIABLE_LIST(self):
        if not self.robust:
            return [[self.HOM, "t", "c"]]
        else:
            base = [self.HOM, "t", "c"]
            return [
                base,
                base + ["w_0"],
                base + ["z_0"],
                base + ["w_0", "w_1"],
                base + ["w_0", "z_0"],
                base + ["z_0", "z_1"],
                # base + ["w_0", "w_1", "z_0"],
                # base + ["w_0", "w_1", "z_0", "z_1"],
            ]

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(
        self,
        n_outliers=0,
        level="no",
        param_level="no",
        d=2,
        n_landmarks=3,
        variable_list=None,
        robust=False,
        beta=BETA,
    ):
        """RobustPoseLifter is a general class for point-to-point, point-to-line, and point-to-plane registration problems,
        with starndard or robust loss functions.

        The goal is to regress an unknown pose based on extrinsic measurements.

        See class:`~popcor.examples.WahbaLifter` for point-to-point registration and :class:`~popcor.examples.MonoLifter`) for point-to-line registration.

        Implemented lifting functions are:

            - xwT: :math:`x \\otimes w`
            - xxT: :math:`x \\otimes x`
        """
        self.beta = beta
        self.n_landmarks = n_landmarks

        self.robust = robust
        self.level = level
        if variable_list == "all":
            variable_list = self.get_all_variables()
        # elif variable_list is None:
        #    self.variable_list = self.VARIABLE_LIST

        if not robust:
            assert level == "no"

        self.landmarks_ = None  # will be initialized later
        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
            variable_list=variable_list,
            n_outliers=n_outliers,
            robust=robust,
        )

    def penalty(self, t, rho=PENALTY_RHO, u=PENALTY_U):
        import autograd.numpy as anp

        try:
            return anp.sum(  # type: ignore
                [rho * u * anp.log10(1 + anp.exp(hi / u)) for hi in self.h_list(t)]  # type: ignore
            )
        except RuntimeWarning:
            u = PENALTY_U * 0.1
            return anp.sum(  # type: ignore
                [rho * u * anp.log10(1 + anp.exp(hi / u)) for hi in self.h_list(t)]  # type: ignore
            )

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        var_dict = {self.HOM: 1, "t": self.d, "c": self.d**2}
        if not self.robust:
            return var_dict

        n = self.d**2 + self.d
        if self.level == "xwT":
            for i in range(self.n_landmarks):
                var_dict.update({f"w_{i}": 1, f"z_{i}": n})
        elif self.level == "xxT":
            var_dict.update({f"w_{i}": 1 for i in range(self.n_landmarks)})
            var_dict.update({"z_0": n**2})
        return var_dict

    @property
    def param_dict(self):
        return self.param_dict_landmarks

    def get_all_variables(self):
        all_variables = [self.HOM, "t", "c"]
        if self.robust:
            if self.level == "xxT":
                all_variables += [f"w_{i}" for i in range(self.n_landmarks)]
                all_variables += ["z_0"]
            elif self.level == "xwT":
                for i in range(self.n_landmarks):
                    all_variables += [f"w_{i}", f"z_{i}"]
        variable_list = [all_variables]
        return variable_list

    def sample_theta(self):
        """Generate a random new feasible point."""

        # make sure random pose is looking at world centre (where landmarks are)
        success = False
        i = 0
        while not success:
            pc_cw = self.get_random_position()
            success = np.all(np.array(self.h_list(pc_cw)) <= 0)
            if success:
                break
            i += 1
            if i >= N_TRYS:
                raise ValueError("didn't find valid initialization")

        if self.d == 2:
            angle = np.random.uniform(0, 2 * np.pi)
            C = R.from_euler("z", angle).as_matrix()[:2, :2]
        else:
            C = R.random().as_matrix()
        theta_x = get_theta_from_C_r(C, pc_cw)

        if self.robust:
            outlier_index = np.random.choice(
                self.n_landmarks, replace=False, size=self.n_outliers
            )
            w = np.ones(self.n_landmarks)
            w[outlier_index] = -1
            return np.hstack([theta_x, w])
        else:
            return theta_x

    @property
    def landmarks(self):
        if self.landmarks_ is None:
            self.landmarks_ = np.random.normal(
                loc=0, scale=1.0, size=(self.n_landmarks, self.d)
            )
        return self.landmarks_

    def sample_parameters(self, theta=None):
        if self.parameters_ is None:
            return self.sample_parameters_landmarks(self.landmarks)
        landmarks = np.random.normal(loc=0, scale=1.0, size=(self.n_landmarks, self.d))
        return self.sample_parameters_landmarks(landmarks)

    def get_x(self, theta=None, var_subset=None, parameters=None) -> np.ndarray:
        """Get the lifted vector x given theta and parameters."""
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict.keys()
        if self.robust:
            theta_here = theta[: -self.n_landmarks]
        else:
            theta_here = theta

        # RT below is R_cw. (c=camera, w=world)
        RT, t = get_C_r_from_theta(theta_here, self.d)
        R = RT.T

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            elif key == "t":
                x_data += list(t)
            elif key == "c":
                x_data += list(R.flatten("C"))
            elif "w" in key:
                j = int(key.split("_")[-1])
                w_j = theta[-self.n_landmarks + j]
                x_data.append(w_j)
            elif (self.level == "xxT") and (key == "z_0"):
                x_vec = list(get_theta_from_C_r(R, t))
                x_data += list(np.kron(x_vec, x_vec).flatten())
            elif (self.level == "xwT") and ("z" in key):
                j = int(key.split("_")[-1])
                w_j = theta[-self.n_landmarks + j]
                x_vec = get_theta_from_C_r(R, t)
                x_data += list(x_vec * w_j)
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_outlier_index(self):
        if self.robust:
            return np.where(self.theta[-self.n_landmarks :] == -1)[0]
        else:
            return []

    def get_theta(self, x):
        # t, vec(R), w
        t = x[: self.d]
        RT = x[self.d : self.d + self.d**2].reshape(self.d, self.d, order="C")
        R = RT.T
        w = x[self.d + self.d**2 : self.d + self.d**2 : self.n_landmarks]
        return np.hstack([t, R.flatten("C"), w])

    def get_error(self, theta_hat, error_type="MSE", atol=1e-10):
        """
        :param atol: tolerance for sanity check of C'C=I constraint.
        """

        theta_hat_pose = theta_hat[: self.d + self.d**2]
        theta_gt_pose = self.theta[: self.d + self.d**2]
        return get_pose_errors_from_theta(
            theta_hat_pose, theta_gt_pose, self.d, atol=atol
        )[error_type]

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        if self.robust:
            theta = deepcopy(self.theta[: self.d + self.d**2])
            C, r = get_C_r_from_theta(theta, self.d)
            theta_noisy = get_noisy_pose(C, r, delta=delta)
            theta_w = self.theta[self.d + self.d**2 :]
            return np.r_[theta_noisy, theta_w]
        else:
            C, r = get_C_r_from_theta(self.theta, self.d)
            theta_noisy = get_noisy_pose(C, r, delta=delta)
            return theta_noisy

    def get_cost(self, theta, y):
        assert y is not None
        if self.robust:
            x = theta[: -self.n_landmarks]
            w = theta[-self.n_landmarks :]
            assert np.all(w**2 == 1.0)
        else:
            x = theta

        R, t = get_C_r_from_theta(x, self.d)

        cost = 0
        for i in range(self.n_landmarks):
            res = self.residual_sq(R, t, self.landmarks[i], y[i])
            if self.robust:
                cost += (1 + w[i]) / self.beta**2 * res + 1 - w[i]
            else:
                cost += res
        return 0.5 * cost

    def local_solver(
        self,
        t0,
        y: np.ndarray | None,
        verbose=False,
        method=METHOD,
        solver_kwargs=SOLVER_KWARGS,
    ):
        assert y is not None
        import pymanopt
        from pymanopt.manifolds import Euclidean, Product, SpecialOrthogonalGroup

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

        # We assume that we know w! If we wanted to solve for w too we would need
        # IRLS or similar. Since we just care about getting the global solution
        # with a local sovler that's not necessary.
        if self.robust:
            w = self.theta[-self.n_landmarks :]

        manifold = Product((SpecialOrthogonalGroup(self.d, k=1), Euclidean(self.d)))

        @pymanopt.function.autograd(manifold)
        def cost(R, t):
            cost = 0
            for i in range(self.n_landmarks):
                residual = self.residual_sq(R, t, self.landmarks[i], y[i])
                if self.robust:
                    cost += (1 + w[i]) / self.beta**2 * residual + 1 - w[i]
                else:
                    cost += residual
            return 0.5 * cost + self.penalty(t)

        @pymanopt.function.autograd(manifold)
        def euclidean_gradient_unused(R, t):
            grad_R = np.zeros(R.shape)
            grad_t = np.zeros(t.shape)
            for i in range(self.n_landmarks):
                Wi = np.eye(self.d) - np.outer(y[i], y[i])
                # residual = (R @ pi + t).T @ Wi @ (R @ pi + t)
                term = self.term_in_norm(R, t, self.landmarks[i], y[i])
                if self.robust:
                    grad_R += (
                        2
                        * w[i]
                        / self.beta**2
                        * np.outer(Wi.T @ term, self.landmarks[i])
                    )
                    grad_t += 2 * w[i] / self.beta**2 * Wi.T @ term
                else:
                    grad_R += np.outer(Wi.T @ term, self.landmarks[i])
                    grad_t += Wi.T @ term
            return grad_R, grad_t

        euclidean_gradient = None
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient
        )
        optimizer = Optimizer(**solver_kwargs)

        R_0, t_0 = get_C_r_from_theta(t0[: self.d + self.d**2], self.d)
        res = optimizer.run(problem, initial_point=(R_0, t_0))
        R, t = res.point

        if verbose:
            print("local solver sanity check:")
            print("final penalty:", self.penalty(t))
        w = self.theta[-self.n_landmarks :]
        for i in range(self.n_landmarks):
            residual = self.residual_sq(R, t, self.landmarks[i], y[i])
            if w[i] == -1:
                if verbose:
                    print(f"outlier residual: {residual:.4e}")
                assert (
                    residual > self.beta
                ), f"outlier residual too small: {residual} <= {self.beta}"
            else:
                if verbose:
                    print(f"inlier residual: {residual:.4e}")
                assert (
                    residual < self.beta
                ), f"inlier residual too large: {residual} > {self.beta}"
        if verbose:
            print("qcqp cost:", res.cost)

        if self.robust:
            theta_hat = np.r_[get_theta_from_C_r(R, t), w]
        else:
            theta_hat = get_theta_from_C_r(R, t)

        cost_penalized = res.cost
        if self.robust:
            pen = self.penalty(t)
            if abs(res.cost) > 1e-10:
                assert abs(pen) / res.cost <= 1e-1, (pen, res.cost)
            cost_penalized -= pen

        success = ("min step_size" in res.stopping_criterion) or (
            "min grad norm" in res.stopping_criterion
        )
        info = {
            "success": success,
            "msg": res.stopping_criterion,
            "cost": cost_penalized,
        }
        if success:
            return theta_hat, info, cost_penalized
        else:
            return None, info, cost_penalized

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
            # enforce diagonal == 1 (R'R=I)
            for i in range(self.d):
                Ei = np.zeros((self.d, self.d))
                Ei[i, i] = 1.0
                constraint = np.kron(Ei, np.eye(self.d))
                Ai = PolyMatrix(symmetric=True)
                Ai["c", "c"] = constraint
                Ai[self.HOM, self.HOM] = -1
                self.test_and_add(A_list, Ai, output_poly=output_poly)

            # enforce off-diagonal == 0 (R'R=I)
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, j] = 1.0
                    Ei[j, i] = 1.0
                    constraint = np.kron(Ei, np.eye(self.d))
                    Ai = PolyMatrix(symmetric=True)
                    Ai["c", "c"] = constraint
                    self.test_and_add(A_list, Ai, output_poly=output_poly)
        if add_redundant and ("c" in var_dict):
            # enforce diagonal == 1 (RR'=I)
            for i in range(self.d):
                Ei = np.zeros((self.d, self.d))
                Ei[i, i] = 1.0
                constraint = np.kron(np.eye(self.d), Ei)
                Ai = PolyMatrix(symmetric=True)
                Ai["c", "c"] = constraint
                Ai[self.HOM, self.HOM] = -1
                self.test_and_add(A_list, Ai, output_poly=output_poly)

            # enforce off-diagonal == 0 (RR'=I)
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, j] = 1.0
                    Ei[j, i] = 1.0
                    constraint = np.kron(np.eye(self.d), Ei)
                    Ai = PolyMatrix(symmetric=True)
                    Ai["c", "c"] = constraint
                    self.test_and_add(A_list, Ai, output_poly=output_poly)

        if self.robust:
            for key in var_dict:
                if "w" in key:
                    i = key.split("_")[-1]
                    Ai = PolyMatrix(symmetric=True)
                    Ai[self.HOM, self.HOM] = -1.0
                    Ai[f"w_{i}", f"w_{i}"] = 1.0
                    self.test_and_add(A_list, Ai, output_poly=output_poly)

                    # below doesn't hold: w_i*w_j = += 1
                    # for key_other in [k for k in var_dict if (k.startswith("w") and (k!= key))]:
                    #    Ai = PolyMatrix(symmetric=True)
                    #    Ai[self.HOM, self.HOM] = -1.0
                    #    Ai[key, key_other] = 0.5
                    #    self.test_and_add(A_list, Ai, output_poly=output_poly)

                if "z" in key:
                    if self.level == "xwT":
                        i = key.split("_")[-1]
                        """ each z_i equals x * w_i"""

                        for j in range(self.d):
                            Ai = PolyMatrix(symmetric=True)
                            constraint = np.zeros((self.d + self.d**2))
                            constraint[j] = 1.0
                            Ai[self.HOM, f"z_{i}"] = constraint[None, :]
                            constraint = np.zeros((self.d))
                            constraint[j] = -1.0
                            Ai[f"t", f"w_{i}"] = constraint[:, None]
                            self.test_and_add(A_list, Ai, output_poly=output_poly)

                        for j in range(self.d**2):
                            Ai = PolyMatrix(symmetric=True)
                            constraint = np.zeros((self.d + self.d**2))
                            constraint[self.d + j] = 1.0
                            Ai[self.HOM, f"z_{i}"] = constraint[None, :]
                            constraint = np.zeros((self.d**2))
                            constraint[j] = -1.0
                            Ai[f"c", f"w_{i}"] = constraint[:, None]
                            self.test_and_add(A_list, Ai, output_poly=output_poly)
        return A_list

    def get_B_known(self):
        """Get inequality constraints of the form x.T @ B @ x <= 0.
        By default, we always add ||t|| <= MAX_DIST
        """
        B1 = PolyMatrix(symmetric=True)
        B1[self.HOM, self.HOM] = -self.MAX_DIST
        B1["t", "t"] = np.eye(self.d)
        return [B1.get_matrix(self.var_dict)]

    @abstractmethod
    def h_list(self, t):
        """
        Any inequality constraints to enforce, returned as a list [h_1(t), h_2(t), ...]
        We use the convention h_i(t) <= 0.

        By default, we always add |t| <= MAX_DIST
        """
        try:
            import autograd.numpy as anp

            return [anp.sqrt(anp.sum(t[: self.d] ** 2)) - self.MAX_DIST]  # type: ignore
        except ModuleNotFoundError:
            return [np.sqrt(np.sum(t[: self.d] ** 2)) - self.MAX_DIST]

    @abstractmethod
    def get_random_position(self):
        """Generate a new random position. Orientation angles will be drawn uniformly from [0, pi]."""
        return None

    @abstractmethod
    def term_in_norm(self, R, t, pi, ui) -> np.ndarray:
        pass

    @abstractmethod
    def residual_sq(
        self,
        R: np.ndarray,
        t: np.ndarray,
        pi: np.ndarray,
        ui: np.ndarray,
    ) -> float:
        pass
