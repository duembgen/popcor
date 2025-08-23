"""Stereo3DLifter class for 2D stereo localization example."""

import pickle

import numpy as np

from popcor.base_lifters import StereoLifter
from popcor.utils.geometry import get_T, get_theta_from_T
from popcor.utils.stereo3d_problem import _cost, local_solver


def change_dimensions(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert landmarks and measurements to solver-friendly shapes."""
    a = np.asarray(a)
    p_w = np.hstack((a, np.ones((a.shape[0], 1), dtype=a.dtype)))[:, :, None]
    y_mat = np.asarray(y).reshape((a.shape[0], -1))
    return p_w, y_mat[:, :, None]


GTOL: float = 1e-6


class Stereo3DLifter(StereoLifter):
    """Stereo-camera localization in 3D.

    Analogously to :class:`Stereo2DLifter`, we minimize the following cost function:

    .. math::
        f(\\theta) = \\sum_{j=0}^{n} (u_j - M q_j / q_j[2])^2

    where

    - :math:`p_j` are known landmarks (in homogeneous coordinates),
    - :math:`u_j` are pixel measurements (four elements: two pixel coordinates in the left image and two in the right image),
    - :math:`q_j = T(\\theta) p_j` are the (homogeneous) coordinates of landmark j in the (unknown) camera frame, parameterized by :math:`T(\\theta)`, and
    - :math:`M` is the stereo camera calibration matrix. Here, it is given by

    .. math::

        \\begin{bmatrix}
            f_u & 0 & c_u & \\frac{b f_u}{2} \\\\
            0   & f_v & c_v & 0 \\\\
            f_u & 0 & c_u & -\\frac{b f_u}{2} \\\\
            0   & f_v & c_v & 0 \\\\
        \\end{bmatrix}

    where :math:`f_u, f_v` are horizontal and vertical focal lengths, :math:`c_u,c_v` are image center points in pixels and :math:`b` is the camera baseline.

    This example is treated in more detail in `this paper <https://arxiv.org/abs/2308.05783>`_.
    """

    def __init__(
        self,
        n_landmarks: int,
        level: str = "no",
        param_level: str = "no",
        variable_list: list | None = None,
    ):
        # Pre-allocate W as repeated identity matrices; repeat is slightly faster than building a list.
        self.W: np.ndarray = np.repeat(np.eye(4)[None, :, :], n_landmarks, axis=0)

        # Precompute the M_matrix once for efficiency.
        f_u: float = 484.5
        f_v: float = 484.5
        c_u: float = 322
        c_v: float = 247
        b: float = 0.24
        self._M_matrix: np.ndarray = np.array(
            [
                [f_u, 0, c_u, f_u * b / 2],
                [0, f_v, c_v, 0],
                [f_u, 0, c_u, -f_u * b / 2],
                [0, f_v, c_v, 0],
            ]
        )

        super().__init__(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            d=3,
            variable_list=variable_list,
        )

    @property
    def M_matrix(self) -> np.ndarray:
        """Stereo camera calibration matrix used by the cost and solver."""
        return self._M_matrix

    @staticmethod
    def from_file(fname: str) -> "Stereo3DLifter":
        """Load a Stereo3DLifter instance state from a file created by to_file."""
        with open(fname, "rb") as f:
            y_, landmarks, theta, level, param_level, variable_list = pickle.load(f)
        lifter = Stereo3DLifter(
            n_landmarks=landmarks.shape[0],
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
        lifter.y_ = y_
        lifter.landmarks_ = landmarks
        lifter.parameters = np.r_[1, landmarks.flatten()]
        lifter.theta = theta
        return lifter

    def to_file(self, fname: str) -> None:
        """Serialize the lifter's minimal state to a file for later restoration."""
        with open(fname, "wb") as f:
            pickle.dump(
                (
                    self.y_,
                    self.landmarks,
                    self.theta,
                    self.level,
                    self.param_level,
                    self.variable_list,
                ),
                f,
            )

    def get_cost(
        self,
        theta: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
    ) -> float:
        """Evaluate the reprojection cost for a given parameter vector theta."""
        if W is None:
            W = self.W
        a = self.landmarks

        p_w, y = change_dimensions(a, y)
        T = get_T(theta=theta, d=3)

        cost = _cost(p_w=p_w, y=y, T=T, M=self.M_matrix, W=W)
        if StereoLifter.NORMALIZE:
            return cost / (self.n_landmarks * self.d)
        else:
            return cost

    def local_solver(
        self,
        t0: np.ndarray,
        y: np.ndarray,
        W: np.ndarray | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray | None, dict, float]:
        """Run the local solver starting from initial pose t0 and measurements y."""
        if W is None:
            W = self.W

        a = self.landmarks
        p_w, y = change_dimensions(a, y)
        T_init = get_T(theta=t0, d=3)

        info, T_hat, cost = local_solver(
            T_init=T_init,
            y=y,
            p_w=p_w,
            W=W,
            M=self.M_matrix,
            log=False,
            gtol=GTOL,
            min_update_norm=-1,  # makes this inactive
        )

        if verbose:
            print("Stereo3D local solver:", info.get("msg", ""))

        if StereoLifter.NORMALIZE:
            cost /= self.n_landmarks * self.d

        x_hat = get_theta_from_T(T_hat)
        x = self.get_x(theta=x_hat)
        Q = self.get_Q_from_y(y[:, :, 0])
        cost_Q = x.T @ Q @ x
        if abs(cost) > 1e-10:
            if not (abs(cost_Q - cost) / cost < 1e-8):
                print(f"Warning, cost not equal {cost_Q:.2e} {cost:.2e}")

        info["cost"] = cost

        if info.get("success", False):
            return x_hat, info, cost
        else:
            return None, info, cost


if __name__ == "__main__":
    lifter = Stereo3DLifter(n_landmarks=4)
