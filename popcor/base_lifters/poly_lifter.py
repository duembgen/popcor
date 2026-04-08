import numpy as np

from .state_lifter import StateLifter


class PolyLifter(StateLifter):
    def __init__(self, degree, param_level="no"):
        """Simple univariate polynomial lifter, mostly for testing and pedagogical purposes."""
        self.degree = degree
        super().__init__(d=1, param_level=param_level)

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {self.HOM: 1, "t": 1}
            self.var_dict_.update({f"z{i}": 1 for i in range(self.M)})
        return self.var_dict_

    @property
    def M(self):
        return self.degree // 2 - 1

    def sample_theta(self):
        return np.random.rand(1)

    def get_error(self, theta_hat, error_type="MSE", *args, **kwargs):
        if error_type == "MSE":
            return float((self.theta - theta_hat) ** 2)
        else:
            raise ValueError(f"Unknown error type: {error_type}")

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if theta is None:
            theta = self.theta
        return np.array([theta**i for i in range(self.degree // 2 + 1)])

    def get_cost(self, theta, y=None) -> float:
        Q = self.get_Q()
        x = self.get_x(theta).flatten()
        return float(x.T @ Q @ x)

    def get_hess(self, y=None):
        raise NotImplementedError

    def local_solver(self, t0, y=None, *args, **kwargs):
        from scipy.optimize import minimize

        sol = minimize(self.get_cost, t0)
        info = {"success": sol.success, "cost": sol.fun}
        return sol.x, info, sol.fun

    def plot(self, thetas=None, label=None, estimates=None):
        from popcor.utils.plotting_tools_poly import coordinate_system

        fig, ax = coordinate_system()

        # Handle the case where estimates is provided but thetas is not
        if thetas is None and estimates is None:
            raise ValueError("Either thetas or estimates must be provided")

        if thetas is not None:
            ys = [self.get_cost(t) for t in thetas]
            ax.plot(thetas, ys, label=label)
            ymin = min(-max(ys) / 3, min(ys))
            ax.set_ylim(ymin, max(ys))

        # Plot estimates if provided
        if estimates is not None:
            for est_label, theta_est in estimates.items():
                cost_est = self.get_cost(theta_est)
                ax.scatter(
                    [theta_est],
                    [cost_est],
                    label=est_label,
                    s=100,
                    marker="x",
                    linewidth=2,
                )

        return fig, ax

    def __repr__(self):
        return f"poly{self.degree}"
