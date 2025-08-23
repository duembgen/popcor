"""MonoLifter class for robust pose registration with monocular measurements."""

from copy import deepcopy

import autograd.numpy as anp
import numpy as np
from poly_matrix.poly_matrix import PolyMatrix

from popcor.base_lifters import RobustPoseLifter
from popcor.utils.geometry import get_C_r_from_theta
from popcor.utils.plotting_tools import plot_frame

FOV = np.pi / 2  # camera field of view
N_TRYS = 10

# TODO(FD) for some reason this is not required as opposed to what is stated in Heng's paper
# and it currently breaks tightness (might be a bug in my implementation though)
USE_INEQ = False

NORMALIZE = False


class MonoLifter(RobustPoseLifter):
    """Robust pose lifter for monocular point-to-line registration.

    Doc under construction.

    In the meantime, this example is treated in more detail in `this paper <https://arxiv.org/abs/2308.05783>`_,
    under the name "PLR".
    """

    NOISE: float = 1e-3  # inlier noise
    NOISE_OUT: float = 0.1  # outlier noise

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def TIGHTNESS(self) -> str:
        return "cost" if self.robust else "rank"

    def h_list(self, t: np.ndarray, *args, **kwargs) -> list:
        """
        Returns constraints h_j(t) <= 0:
        - norm(t) <= 10 (default)
        - tan(a/2)*t3 >= sqrt(t1**2 + t2**2)
        """
        default = super().h_list(t, *args, **kwargs)
        return default + [
            anp.sum(t[:-1] ** 2) - anp.tan(FOV / 2) ** 2 * t[-1] ** 2,  # type: ignore
            -t[-1],
        ]

    def get_random_position(self, *args, **kwargs) -> np.ndarray:
        pc_cw = np.random.rand(self.d) * 0.1
        pc_cw[self.d - 1] = np.random.uniform(1, self.MAX_DIST)
        return pc_cw

    def get_B_known(self, *args, **kwargs) -> list:
        """Get inequality constraints of the form x.T @ B @ x <= 0."""
        if not USE_INEQ:
            return []
        default = super().get_B_known(*args, **kwargs)
        B3 = PolyMatrix(symmetric=True)
        constraint = np.zeros((self.d, self.d))
        constraint[range(self.d - 1), range(self.d - 1)] = 1.0
        constraint[self.d - 1, self.d - 1] = -np.tan(FOV / 2) ** 2
        B3["t", "t"] = constraint
        constraint = np.zeros(self.d)
        constraint[self.d - 1] = -1
        B2 = PolyMatrix(symmetric=True)
        B2[self.HOM, "t"] = constraint[None, :]
        return default + [
            B2.get_matrix(self.var_dict),
            B3.get_matrix(self.var_dict),
        ]

    def term_in_norm(
        self,
        R: np.ndarray,
        t: np.ndarray,
        pi: np.ndarray,
        ui: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return R @ pi + t

    def residual_sq(
        self,
        R: np.ndarray,
        t: np.ndarray,
        pi: np.ndarray,
        ui: np.ndarray,
        *args,
        **kwargs,
    ) -> float | anp.numpy_boxes.ArrayBox:
        W = np.eye(self.d) - np.outer(ui, ui)
        term = self.term_in_norm(R, t, pi, ui, *args, **kwargs)
        if isinstance(term, np.ndarray):
            res = float(term.T @ W @ term)
        else:
            res = term.T @ W @ term

        if NORMALIZE:
            res /= (self.n_landmarks * self.d) ** 2
        return res

    def plot_setup(self, *args, **kwargs) -> tuple:
        if self.d != 2:
            print("Plotting currently only supported for d=2")
            return None, None
        import matplotlib.pylab as plt

        assert self.landmarks is not None
        fig, ax = plt.subplots()
        ax.axis("equal")
        t_wc_w, C_cw = plot_frame(ax, self.theta, label="pose", color="gray", d=2)
        if self.y_ is not None:
            for i in range(self.y_.shape[0]):
                ax.scatter(
                    self.landmarks[i][0],
                    self.landmarks[i][1],
                    color=f"C{i}",
                    label="landmarks",
                )
                ui_c = self.y_[i]
                assert abs(np.linalg.norm(ui_c) - 1.0) < 1e-10
                ax.plot(
                    [t_wc_w[0], self.landmarks[i][0]],
                    [t_wc_w[1], self.landmarks[i][1]],
                    color=f"C{i}",
                    ls=":",
                )
                if C_cw is not None:
                    ui_w = C_cw.T @ ui_c
                    ax.plot(
                        [t_wc_w[0], t_wc_w[0] + ui_w[0]],
                        [t_wc_w[1], t_wc_w[1] + ui_w[1]],
                        color=f"r" if i < self.n_outliers else "g",
                    )
        return fig, ax

    def simulate_y(self, noise: float | None = None) -> np.ndarray:
        if noise is None:
            noise = self.NOISE
        y = np.zeros((self.n_landmarks, self.d))
        theta = self.theta[: self.d + self.d**2]
        outlier_index = self.get_outlier_index()
        R, t = get_C_r_from_theta(theta, self.d)
        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            ui = R @ pi + t
            ui /= ui[self.d - 1]
            if np.tan(FOV / 2) * ui[self.d - 1] < np.sqrt(
                np.sum(ui[: self.d - 1] ** 2)
            ):
                print("warning: inlier not in FOV!!")
            if i in outlier_index:
                success = False
                for _ in range(N_TRYS):
                    ui_test = deepcopy(ui)
                    ui_test[: self.d - 1] += np.random.normal(
                        scale=self.NOISE_OUT / 10.0, loc=self.NOISE_OUT, size=self.d - 1
                    )
                    if np.tan(FOV / 2) * ui_test[self.d - 1] >= np.sqrt(
                        np.sum(ui_test[: self.d - 1] ** 2)
                    ):
                        success = True
                        ui = ui_test
                        break
                if not success:
                    raise ValueError("did not find valid outlier ui")
            else:
                ui[: self.d - 1] += np.random.normal(
                    scale=noise, loc=0, size=self.d - 1
                )
            assert ui[self.d - 1] == 1.0
            ui /= np.linalg.norm(ui)
            y[i] = ui
        self.y_ = y
        return y

    def get_Q(
        self,
        noise: float | None = None,
        output_poly: bool = False,
        use_cliques: list = [],
        *args,
        **kwargs,
    ) -> np.ndarray | PolyMatrix:
        assert self.landmarks is not None, "landmarks must be set before calling get_Q"
        if noise is None:
            noise = self.NOISE
        if self.y_ is None:
            self.y_ = self.simulate_y(noise=noise)
        Q = self.get_Q_from_y(
            self.y_, output_poly=output_poly, use_cliques=use_cliques, *args, **kwargs
        )
        return Q

    def get_Q_from_y(
        self,
        y: np.ndarray,
        output_poly: bool = False,
        use_cliques: list = [],
        *args,
        **kwargs,
    ) -> np.ndarray | PolyMatrix:
        """
        Returns the cost matrix Q for the current y.
        """
        assert (
            self.landmarks is not None
        ), "landmarks must be set before calling get_Q_from_y"
        Q = PolyMatrix(symmetric=True)
        if NORMALIZE:
            norm = (self.n_landmarks * self.d) ** 2
        js = use_cliques if len(use_cliques) else list(range(self.n_landmarks))
        for i in js:
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]
            Wi = np.eye(self.d) - np.outer(ui, ui)
            Qi = Pi.T @ Wi @ Pi
            if NORMALIZE:
                Qi /= norm
            if self.robust:
                Qi /= self.beta**2
                Q[self.HOM, self.HOM] += 1
                Q[self.HOM, f"w_{i}"] += -0.5
                if self.level == "xwT":
                    Q[f"z_{i}", "t"] += 0.5 * Qi[:, : self.d]
                    Q[f"z_{i}", "c"] += 0.5 * Qi[:, self.d :]
                    Q["t", "t"] += Qi[: self.d, : self.d]
                    Q["t", "c"] += Qi[: self.d, self.d :]
                    Q["c", "c"] += Qi[self.d :, self.d :]
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                    Q["t", "t"] += Qi[: self.d, : self.d]
                    Q["t", "c"] += Qi[: self.d, self.d :]
                    Q["c", "c"] += Qi[self.d :, self.d :]
            else:
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]
        if output_poly:
            return 0.5 * Q
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def __repr__(self, *args, **kwargs) -> str:
        appendix = "_robust" if self.robust else ""
        return f"mono_{self.d}d_{self.level}_{self.param_level}{appendix}"
