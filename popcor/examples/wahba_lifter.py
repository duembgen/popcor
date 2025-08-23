"""WahbaLifter class for robust pose registration with point-to-point measurements."""

import autograd.numpy as anp
import numpy as np
import scipy.sparse as sp
from poly_matrix import PolyMatrix

from popcor.base_lifters import RobustPoseLifter
from popcor.utils.geometry import get_C_r_from_theta
from popcor.utils.plotting_tools import plot_frame

N_TRYS: int = 10

# TODO(FD) for some reason this is not required as opposed to what is stated in Heng's paper
# and it currently breaks tightness (might be a bug in my implementation though)
USE_INEQ: bool = False

NORMALIZE: bool = False


class WahbaLifter(RobustPoseLifter):
    """Robust pose lifter for point-to-point registration.

    Doc under construction.

    In the meantime, this example is treated in more detail in `this paper <https://arxiv.org/abs/2308.05783>`_,
    under the name "PPR".

    """

    NOISE: float = 1e-2  # inlier noise
    NOISE_OUT: float = 1.0  # outlier noise

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def h_list(self, t: np.ndarray) -> list:
        """
        Returns constraints h_j(t) <= 0, e.g., norm(t) <= 10 (default).
        """
        default = super().h_list(t)
        return default

    def get_random_position(self) -> np.ndarray:
        """Generates a random position within the allowed distance."""
        return np.random.uniform(
            -0.5 * self.MAX_DIST ** (1 / self.d),
            0.5 * self.MAX_DIST ** (1 / self.d),
            size=self.d,
        )

    def get_B_known(self) -> list:
        """Get inequality constraints of the form x.T @ B @ x <= 0."""
        if not USE_INEQ:
            return []
        default = super().get_B_known()
        return default

    def term_in_norm(
        self, R: np.ndarray, t: np.ndarray, pi: np.ndarray, ui: np.ndarray
    ) -> np.ndarray:
        """Computes the term inside the norm for residual calculation."""
        return R @ pi + t - ui

    def residual_sq(
        self, R: np.ndarray, t: np.ndarray, pi: np.ndarray, ui: np.ndarray
    ) -> float | anp.numpy_boxes.ArrayBox:
        """Computes the squared residual for a landmark measurement."""
        W = np.eye(self.d)
        term = self.term_in_norm(R, t, pi, ui)
        if isinstance(term, np.ndarray):
            res_sq = float(term.T @ W @ term)
        else:
            res_sq = term.T @ W @ term
        if NORMALIZE:
            return res_sq / (self.n_landmarks * self.d) ** 2
        return res_sq

    def plot_setup(self) -> tuple | None:
        """Plots the pose and landmarks setup for d=2."""
        if self.d != 2:
            print("Plotting currently only supported for d=2")
            return None
        import matplotlib.pylab as plt

        fig, ax = plt.subplots()
        ax.axis("equal")
        t_wc_w, C_cw = plot_frame(ax, self.theta, label="pose", color="gray", d=2)

        if self.y_ is not None:
            w = self.theta[-self.n_landmarks :]
            for i in range(self.y_.shape[0]):
                ax.scatter(*self.landmarks[i], color=f"C{i}", label="landmarks")
                t_cpi_c = self.y_[i]
                ax.plot(
                    [t_wc_w[0], self.landmarks[i][0]],
                    [t_wc_w[1], self.landmarks[i][1]],
                    color=f"C{i}",
                    ls=":",
                )
                if C_cw is not None:
                    t_cpi_w = C_cw.T @ t_cpi_c
                    ax.plot(
                        [t_wc_w[0], t_wc_w[0] + t_cpi_w[0]],
                        [t_wc_w[1], t_wc_w[1] + t_cpi_w[1]],
                        color=f"r" if w[i] < 0 else "g",
                    )
        ax.axis("equal")
        return fig, ax

    def simulate_y(self, noise: float | None = None) -> np.ndarray:
        """Simulates landmark measurements with noise and outliers."""
        if noise is None:
            noise = self.NOISE

        theta = self.theta[: self.d + self.d**2]
        outlier_index = self.get_outlier_index()

        y = np.empty((self.n_landmarks, self.d))
        R, t = get_C_r_from_theta(theta, self.d)
        for i in range(self.n_landmarks):
            valid_measurement = False
            for _ in range(N_TRYS):
                outlier = i in outlier_index
                y_i = R @ self.landmarks[i] + t
                if outlier:
                    y_i += np.random.normal(scale=self.NOISE_OUT, loc=0, size=self.d)
                else:
                    y_i += np.random.normal(scale=noise, loc=0, size=self.d)

                residual = self.residual_sq(R, t, self.landmarks[i], y_i)
                if not self.robust:
                    valid_measurement = True
                else:
                    if outlier:
                        valid_measurement = residual > self.beta
                    else:
                        valid_measurement = residual < self.beta
                if valid_measurement:
                    break
            if not valid_measurement and self.robust:
                self.plot_setup()
                raise ValueError("Did not find a valid measurement.")
            y[i] = y_i
        self.y_ = y
        return y

    def get_Q(
        self,
        noise: float | None = None,
        output_poly: bool = False,
        use_cliques: list = [],
    ) -> np.ndarray | PolyMatrix | sp.csr_matrix | sp.csc_matrix:
        """Returns the quadratic cost matrix Q for the current measurements."""
        if noise is None:
            noise = self.NOISE

        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)
        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q

    def get_Q_from_y(
        self,
        y: np.ndarray,
        output_poly: bool = False,
        use_cliques: list = [],
    ) -> np.ndarray | PolyMatrix | sp.csr_matrix | sp.csc_matrix:
        """
        Returns the quadratic cost matrix Q from measurements y.
        Every cost term can be written as:
        (1 + wi)/b^2  r^2(x, zi) + (1 - wi)
        """
        if len(use_cliques):
            js = use_cliques
        else:
            js = list(range(self.n_landmarks))

        from poly_matrix import PolyMatrix

        Q = PolyMatrix(symmetric=True)
        if NORMALIZE:
            norm = (self.n_landmarks * self.d) ** 2

        Wi = np.eye(self.d)
        for i in js:
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]

            Pi_ll = ui.T @ Wi @ ui
            Pi_xl = -(Pi.T @ Wi @ ui)[:, None]
            Qi = Pi.T @ Wi @ Pi
            if NORMALIZE:
                Pi_ll /= norm
                Pi_xl /= norm
                Qi /= norm

            if self.robust:
                Qi /= self.beta**2
                Pi_ll /= self.beta**2
                Pi_xl /= self.beta**2
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]

                Q["t", self.HOM] += Pi_xl[: self.d, :]
                Q["c", self.HOM] += Pi_xl[self.d :, :]
                Q[self.HOM, self.HOM] += (
                    1 + Pi_ll
                )  # 1 from (1 - wi), Pi_ll from first term.
                Q[
                    self.HOM, f"w_{i}"
                ] += -0.5  # from (1 - wi), 0.5 cause on off-diagonal
                if self.level == "xwT":
                    Q[f"z_{i}", "t"] += 0.5 * Qi[:, : self.d]
                    Q[f"z_{i}", "c"] += 0.5 * Qi[:, self.d :]

                    Q[self.HOM, f"w_{i}"] += 0.5 * Pi_ll  # from first term

                    Q[f"z_{i}", self.HOM] += Pi_xl
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]

                    Q["t", f"w_{i}"] += Pi_xl[: self.d, :]
                    Q["c", f"w_{i}"] += Pi_xl[self.d :, :]

                    Q[self.HOM, f"w_{i}"] += 0.5 * Pi_ll
            else:
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]

                Q["t", self.HOM] += Pi_xl[: self.d, :]
                Q["c", self.HOM] += Pi_xl[self.d :, :]
                Q[self.HOM, self.HOM] += Pi_ll  # on diagonal
        if output_poly:
            return 0.5 * Q
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def __repr__(self) -> str:
        appendix = "_robust" if self.robust else ""
        return f"wahba_{self.d}d_{self.level}_{self.param_level}{appendix}"
