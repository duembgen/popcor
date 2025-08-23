import warnings
from abc import ABC, abstractmethod

import autograd.numpy as anp
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .state_lifter import StateLifter

METHOD = "BFGS"
NORMALIZE = True

# TODO(FD): parameters below are not all equivalent.
SOLVER_KWARGS = {
    "BFGS": dict(gtol=1e-6, xrtol=1e-10),  # relative step size
    "Nelder-Mead": dict(xatol=1e-10),  # absolute step size
    "Powell": dict(ftol=1e-6, xtol=1e-10),
    "TNC": dict(gtol=1e-6, xtol=1e-10),
}


MAX_TRIALS = 10  # number of trials to find a valid sample


class RangeOnlyLifter(StateLifter, ABC):
    """Range-only localization base class, in 2D or 3D.

    This is base class for different flavors of the range-only localization problem,
    where the goal is to estimate positions from distance measurements to fixed landmarks.

    See :class:`~popcor.examples.RangeOnlyNsqLifter` and :class:`~popcor.examples.RangeOnlySqLifter`
    for concrete implementations.
    """

    NOISE = 1e-2  # std deviation of distance noise
    SCALE = 2.0  # size of the region of intereist: [0, SCALE]^d
    MIN_DIST = 1e-2  # minimum distance between landmarks and positions

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
        self.n_positions = n_positions
        self.n_landmarks = n_landmarks
        self.landmarks_ = None  # will be set later

        if W is not None:
            assert W.shape == (n_landmarks, n_positions)
            self.W = W
        else:
            self.W = np.ones((n_positions, n_landmarks))
        self.y_ = None

        if variable_list == "all":
            variable_list = self.get_all_variables()

        super().__init__(
            level=level, d=d, variable_list=variable_list, param_level=param_level
        )

    @staticmethod
    def create_bad_fixed(n_positions, n_landmarks, d=2):
        assert n_positions == 1
        landmarks = np.random.rand(n_landmarks, d)
        landmarks[:, 1] *= 0.1  # align landmarks along X.
        theta = np.array([[0.2, 0.3]]).reshape((1, -1))
        return landmarks, theta

    @staticmethod
    def create_bad(n_positions, n_landmarks, d=2):
        # create landmarks that are roughly in a subspace of dimension d-1
        landmarks = np.hstack(  # type: ignore
            [
                np.random.rand(n_landmarks, d - 1) * RangeOnlyLifter.SCALE,
                np.random.rand(n_landmarks, 1) * 0.3 + RangeOnlyLifter.SCALE / 2.0,
            ]
        )
        theta = np.hstack(
            [
                np.random.rand(n_positions, d - 1) * RangeOnlyLifter.SCALE,
                np.max(landmarks[:, -1]) + np.random.rand(n_positions, 1),
            ]
        )
        return landmarks, theta

    @staticmethod
    def create_good(n_positions, n_landmarks, d=2):
        landmarks = RangeOnlyLifter.sample_landmarks_filling_space(n_landmarks, d)
        theta = np.random.uniform(
            [np.min(landmarks, axis=0)],
            [np.max(landmarks, axis=0)],
            size=(n_positions, d),
        )
        return landmarks, theta

    @staticmethod
    def sample_landmarks_filling_space(n_landmarks, d=2):
        landmarks = np.random.rand(n_landmarks, d)
        landmarks = (landmarks - np.min(landmarks, axis=0)) / (
            np.max(landmarks, axis=0) - np.min(landmarks, axis=0)
        )
        # remove landmarks a bit from border for plotting reasons
        landmarks = (
            (landmarks + RangeOnlyLifter.SCALE * 0.05) * RangeOnlyLifter.SCALE * 0.9
        )
        return landmarks

    @property
    def landmarks(self):
        landmarks = RangeOnlyLifter.sample_landmarks_filling_space(
            self.n_landmarks, self.d
        )
        if self.landmarks_ is None:
            self.landmarks_ = landmarks
        return self.landmarks_

    @landmarks.setter
    def landmarks(self, landmarks):
        assert landmarks.shape == (self.n_landmarks, self.d)
        self.landmarks_ = landmarks

    @property
    def VARIABLE_LIST(self):
        return [
            [self.HOM, "x_0"],
            [self.HOM, "x_0", "z_0"],
            [self.HOM, "x_0", "z_0", "z_1"],
            [self.HOM, "x_0", "x_1", "z_0", "z_1"],
        ]

    @abstractmethod
    def get_all_variables(self) -> list:
        return []

    @abstractmethod
    def get_cost(self, theta, y, sub_idx=None, ad=False) -> float:
        return 0.0

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        assert self.landmarks is not None, "landmarks must be set before sampling"

        if delta == 0:
            return self.theta
        else:
            bbox_max = np.max(self.landmarks, axis=0) * 2
            bbox_min = np.min(self.landmarks, axis=0) * 2
            pos = (
                np.random.rand(self.n_positions, self.d)
                * (bbox_max - bbox_min)[None, :]
                + bbox_min[None, :]
            )
            return pos.flatten()

    def sample_parameters(self, theta=None):
        landmarks = np.random.rand(self.n_landmarks, self.d)
        return self.sample_parameters_landmarks(landmarks)

    def overwrite_theta(self, theta):
        """To bypass "theta can only be set once" check."""
        assert theta.shape == (self.n_positions, self.d)
        self.theta_ = theta

    def sample_theta(self):
        """Make sure we do not sample too close to landmarks (if distance is zero we get numerical problems)."""
        samples = np.empty((self.n_positions, self.d))
        for i in range(self.n_positions):
            for j in range(MAX_TRIALS):
                sample = (
                    np.random.rand(self.landmarks.shape[1])
                ) * self.SCALE  # between 0 and SCALE
                distances = np.linalg.norm(sample[None, :] - self.landmarks, axis=1)
                if np.all(distances > self.MIN_DIST):
                    break
                if j == MAX_TRIALS - 1:
                    warnings.warn(
                        f"Did not find valid sample in {MAX_TRIALS} trials. Using last sample with distances {distances.round(4)}.",
                        UserWarning,
                    )
            samples[i, :] = sample
        return samples.flatten("C")

    def get_residuals(self, t, y, squared=True, ad=False):
        positions = t.reshape((-1, self.d))
        sum = anp.sum if ad else np.sum  # type: ignore
        norm = anp.linalg.norm if ad else np.linalg.norm  # type: ignore

        if squared:
            y_current = sum(
                (self.landmarks[None, :, :] - positions[:, None, :]) ** 2, axis=2
            )
            return self.W * (y**2 - y_current)
        else:
            y_current = norm(
                (self.landmarks[None, :, :] - positions[:, None, :]), axis=2
            )
            return self.W * (y - y_current)

    def get_cost_from_res(self, residuals, sub_idx, ad=False):
        """
        Get cost for given positions, landmarks and noise.

        :param t: flattened positions of length Nd
        :param y: N x K distance measurements
        """
        if ad:
            if sub_idx is None:
                cost = anp.sum(residuals**2)  # type: ignore
            else:
                cost = anp.sum(residuals[sub_idx] ** 2)  # type: ignore
            if NORMALIZE:
                cost /= anp.sum(self.W > 0)  # type: ignore
        else:
            if sub_idx is None:
                cost = np.sum(residuals**2)
            else:
                cost = np.sum(residuals[sub_idx] ** 2)
            if NORMALIZE:
                cost /= np.sum(self.W > 0)
        return cost

    def simulate_y(self, noise: float | None = None, squared: bool = True):
        assert self.landmarks is not None
        # N x K matrix
        if noise is None:
            noise = self.NOISE
        positions = self.theta.reshape(self.n_positions, -1)
        y_gt = np.linalg.norm(
            self.landmarks[None, :, :] - positions[:, None, :], axis=2
        )
        if squared:
            return y_gt**2 + np.random.normal(loc=0, scale=noise, size=y_gt.shape)
        else:
            return y_gt + np.random.normal(loc=0, scale=noise, size=y_gt.shape)

    def get_Q(self, noise: float | None = None, output_poly: bool = False) -> tuple:
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)
        Q = self.get_Q_from_y(self.y_, output_poly=output_poly)

        # DEBUGGING
        x = self.get_x()
        cost1 = x.T @ Q @ x
        cost3 = self.get_cost(self.theta, self.y_)
        assert abs(cost1 - cost3) < 1e-10
        return Q

    def get_sub_idx_x(self, sub_idx, add_z=True):
        sub_idx_x = [0]
        for idx in sub_idx:
            sub_idx_x += [1 + idx * self.d + d for d in range(self.d)]
        if not add_z:
            return sub_idx_x
        for idx in sub_idx:
            sub_idx_x += [
                1 + self.n_positions * self.d + idx * self.size_z + d
                for d in range(self.size_z)
            ]
        return sub_idx_x

    def get_theta(self, x):
        assert abs(x[0] - 1.0) > 1e-10
        # below is if we have order x_1, z_1, x_2, z_2, ...
        # x.reshape((self.n_positions, -1))[:, : self.d].flatten("C")
        # below is if we have order x_1, x_2, ..., z_1, z_2, ...
        return x[: self.n_positions * self.d]

    def get_error(self, theta_hat, error_type="rmse", *args, **kwargs):
        assert np.ndim(theta_hat) <= 1
        theta_gt = self.theta if np.ndim(self.theta) <= 1 else self.theta.flatten()
        if error_type == "rmse":
            return np.sqrt(np.mean((theta_gt - theta_hat) ** 2))
        elif error_type == "mse":
            return np.mean((theta_gt - theta_hat) ** 2)
        else:
            raise ValueError(f"Unkwnon error_type {error_type}")

    def local_solver(
        self,
        t0,
        y,
        verbose=False,
        method="BFGS",
        solver_kwargs=SOLVER_KWARGS,
    ):
        """
        :param t_init: (positions, landmarks) tuple
        """

        use_autograd = False
        if self.get_grad(t0, y) is None:
            use_autograd = True

        if use_autograd:
            from autograd import grad  # , hessian

            def fun(x):
                return self.get_cost(theta=x, y=y, ad=True)

            # TODO(FD): split problem into individual points.
            options = solver_kwargs[method]
            options["disp"] = verbose
            sol = minimize(
                fun,
                x0=t0,
                args=y,
                jac=grad(fun),  # type: ignore
                # hess=hessian(fun), not used by any solvers
                method=method,
                options=options,
            )
        else:
            # TODO(FD): split problem into individual points.
            options = solver_kwargs[method]
            options["disp"] = verbose
            sol = minimize(
                self.get_cost,
                x0=t0,
                args=y,
                jac=self.get_grad,
                # hess=self.get_hess, not used by any solvers.
                method=method,
                options=options,
            )

        info = {}
        info["success"] = sol.success
        info["msg"] = sol.message + f"(# iterations: {sol.nit})"
        if sol.success:
            that = sol.x
            rel_error = self.get_cost(that, y) - self.get_cost(sol.x, y)
            assert abs(rel_error) < 1e-10, rel_error
            residuals = self.get_residuals(that, y)
            cost = sol.fun
            info["max res"] = np.max(np.abs(residuals))
            hess = self.get_hess(that, y)
            assert isinstance(hess, sp.csc_matrix)
            eigs = np.linalg.eigvalsh(hess.toarray())
            info["cond Hess"] = eigs[-1] / eigs[0]
        else:
            that = cost = None
            info["max res"] = None
            info["cond Hess"] = None
        info["cost"] = cost
        return that, info, cost

    def plot(self, y=None, xlims=[0, 2], ylims=[0, 2], ax=None, estimates={}):
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(5, 5)
        else:
            fig = plt.gcf()

        if np.ndim(self.theta) < 2:
            theta_gt = self.theta.reshape((self.n_positions, self.d))

        ax.scatter(*self.landmarks[:, :2].T, color="k", marker="+", label="landmarks")
        ax.scatter(*theta_gt[:, :2].T, color="C0", marker="o")
        for label, theta_est in estimates.items():
            if np.ndim(theta_est) < 2:
                theta_est = theta_est.reshape((self.n_positions, self.d))
            ax.scatter(*theta_est[:, :2].T, marker="x", label=label)

        im = None
        if y is not None:
            xs = np.linspace(xlims[0], xlims[1], 100)
            ys = np.linspace(ylims[0], ylims[1], 100)
            xx, yy = np.meshgrid(xs, ys)
            zz = [
                self.get_cost(theta=np.array([xi, yi])[None, :], y=y)
                for xi, yi in zip(xx.flatten(), yy.flatten())
            ]
            im = ax.pcolormesh(
                xx,
                yy,
                np.reshape(zz, xx.shape),
                norm="log",
                alpha=0.5,
                vmin=1e-5,
                vmax=1,
            )
        ax.set_aspect("equal")
        return fig, ax, im

    @property
    @abstractmethod
    def size_z(self) -> int:
        return 1

    @property
    def param_dict(self):
        return self.param_dict_landmarks

    @property
    def N(self):
        return self.n_positions * self.d

    @property
    def M(self):
        return self.n_positions * self.size_z

    def __repr__(self):
        return f"rangeonlyloc{self.d}d_{self.level}"
