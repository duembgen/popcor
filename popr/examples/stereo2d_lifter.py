# import autograd.numpy as np
import numpy as np

from popr.base_lifters import StereoLifter
from popr.utils.geometry import convert_phi_to_theta, convert_theta_to_phi
from popr.utils.stereo2d_problem import _cost, local_solver


def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


GTOL = 1e-6


class Stereo2DLifter(StereoLifter):
    """Stereo-camera localization in 2D.

    We minimize the following cost function:

    .. math::
        f(\\theta) = \\sum_{j=0}^{n} (u_j - M q_j / q_j[1])^2

    where

    - :math:`p_j` are known landmarks (in homogeneous coordinates),
    - :math:`u_j` are pixel measurements (2 elements: one pixel in left "image" and one in right "image"),
    - :math:`q_j = T(\\theta) p_j` are the (homogeneous) coordinates of landmark j in the (unknown) camera frame, parameterized by :math:`T(\\theta)`, and
    - :math:`M` is the stereo camera calibration matrix. Here, it is given by

    .. math::

        \\begin{bmatrix}
            f_u & c_u & \\frac{b f_u}{2} \\\\
            f_v & c_v & -\\frac{b f_v}{2} \\\\
        \\end{bmatrix}

    where :math:`f_u, f_v` are horizontal and vertical focal lengths, :math:`c_u,c_v` are image center points in pixels and :math:`b` is the camera baseline.

    This example is treated in more details in `this paper <https://arxiv.org/abs/2308.05783>`_.
    """

    def __init__(self, n_landmarks, level="no", param_level="no", variable_list=None):
        self.W = np.stack([np.eye(2)] * n_landmarks)

        super().__init__(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            d=2,
            variable_list=variable_list,
        )

    @property
    def M_matrix(self):
        f_u = 484.5
        c_u = 322
        b = 0.24
        return np.array([[f_u, c_u, f_u * b / 2], [f_u, c_u, -f_u * b / 2]])

    def get_cost(self, t, y, W=None):

        if W is None:
            W = self.W
        a = self.landmarks

        phi = convert_theta_to_phi(t)
        p_w, y, phi = change_dimensions(a, y, phi)
        cost = _cost(phi, p_w, y, W, self.M_matrix)
        if StereoLifter.NORMALIZE:
            return cost / (self.n_landmarks * self.d)
        else:
            return cost

    def local_solver(self, t_init, y, W=None, verbose=False, **kwargs):

        if W is None:
            W = self.W
        a = self.landmarks

        init_phi = convert_theta_to_phi(t_init)
        p_w, y, __ = change_dimensions(a, y, init_phi)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=W, init_phi=init_phi, log=verbose, gtol=GTOL
        )
        if StereoLifter.NORMALIZE:
            cost /= self.n_landmarks * self.d
        # cost /= self.n_landmarks * self.d
        theta_hat = convert_phi_to_theta(phi_hat)
        info = {"success": success, "msg": "converged"}
        if success:
            return theta_hat, info, cost
        else:
            return None, info, cost


if __name__ == "__main__":
    lifter = Stereo2DLifter(n_landmarks=3)
