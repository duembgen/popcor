from abc import abstractmethod

import numpy as np

from ._base_class import BaseClass


class StateLifter(BaseClass):
    # sparse hierarchy: define the levels that are implemented
    LEVELS = ["no"]

    # used for AutoTemplate
    VARIABLE_LIST = [["h"]]
    TIGHTNESS = "cost"

    # to be overwritten by inheriting class
    NOISE = 1e-2

    def __init__(
        self,
        level="no",
        param_level="no",
        d=2,
        variable_list=None,
        robust=False,
        n_outliers=0,
        n_parameters=1,
    ):

        # variables that get overwritten upon initialization
        self.parameters_ = None
        self.theta_ = None
        self.var_dict_ = None
        self.y_ = None

        self.robust = robust
        self.n_outliers = n_outliers

        assert level in self.LEVELS
        self.level = level

        if variable_list is not None:
            self.variable_list = variable_list
        else:
            self.variable_list = self.VARIABLE_LIST

        if (param_level != "no") and (n_parameters == 1):
            print("Warning: make sure to give the correct n_parameters for the level.")

        super().__init__(d, param_level, n_parameters)

    # MUST OVERWRITE THESE

    @property
    def var_dict(self):
        raise ValueError("Inheriting class must implement this!")

    @abstractmethod
    def sample_theta(self) -> np.ndarray:
        """Randomly sample a feasible state theta. This function must be
        implemented by the inheriting class."""
        raise NotImplementedError("need to implement sample_theta")

    # MUST OVERWRITE THESE FOR TIGHTNESS CHECKS

    def simulate_y(self, noise: float | None = None) -> np.ndarray:
        """Simulate the measurements y from the current state theta.

        Must provide this funciton if a notion of "noise" and "measurements" exists and shall be used.
        """
        return None

    def get_Q(self, output_poly: bool = False, noise: float | None = None):
        """Construct the cost matrix Q.

        :param noise: set the noise level, if appropriate.
        :param output_poly: if True, return the matrix in PolyMatrix format.

        :returns: the cost matrix as a sparse matrix or PolyMatrix.
        """
        raise NotImplementedError(
            "Need to impelement get_Q in inheriting class if you want to use it."
        )

    def get_Q_from_y(self, y, output_poly: bool = False):
        if y is None:
            return self.get_Q(output_poly=output_poly)
        else:
            raise NotImplementedError(
                "Need to impelement get_Q_from_y in inheriting class if you want to use it."
            )

    def get_A_known(
        self,
        add_redundant: bool = False,
        var_dict: dict | None = None,
        output_poly: bool = False,
    ) -> list:
        """Construct the matrices defining the known equality constraints.

        :param add_redundant: if True, add redundant constraints.
        :param var_dict: if provided, return only the matrices that involve these variables.
        :param output_poly: if True, return the matrices in PolyMatrix format.
        """
        return []

    def get_B_known(self) -> list:
        """Construct the matrices defining the known inequality constraints."""
        return []

    # MUST OVERWRITE THESE FOR ADDING PARAMETERS

    def sample_parameters(self, theta: np.ndarray | None = None) -> dict:
        """Create random set of parameters. By default, there are no parameters
        so this function just returns `{self.HOM: 1.0}`."""
        assert (
            self.param_level == "no"
        ), "Need to overwrite sample_parameters to use level different than 'no'"
        return {self.HOM: 1.0}

    @property
    def param_dict(self):
        assert (
            self.param_level == "no"
        ), "Need to overwrite param_dict to use level different than 'no'"
        return {self.HOM: 1}

    def get_involved_param_dict(self, var_subset):
        """Find which parameters to include, given the current var_subset. Here we implicitly assume
        that each parameter is associated with a variable. This is true for parameters that involve
        substitution variables."""
        keys = [self.HOM]
        for v in var_subset:
            index = v.split("_")
            if len(index) > 1:
                index = int(index[-1])
                key = f"p_{index}"
                if key not in keys:
                    keys.append(key)
        return {k: self.param_dict[k] for k in keys if k in self.param_dict}

    # CAN OPTINALLY OVERWRITE THESE FOR BETTER PERFORMANCE

    def get_grad(self, theta, y=None) -> float:
        raise NotImplementedError("must define get_grad if you want to use it.")

    def get_hess(self, theta, y=None) -> float:
        raise NotImplementedError("must define get_hess if you want to use it.")

    def get_cost(self, theta, y: np.ndarray | None = None) -> float:
        """Compute the cost of the given state theta. This uses the simple form
        x.T @ Q @ x. Consider overwriting this for more efficient computations."""
        print(
            "Warning: using default get_cost, which may be less efficient than a custom one."
        )
        x = self.get_x(theta=theta).flatten("C")
        if y is not None:
            Q = self.get_Q_from_y(y)
        else:
            Q = self.get_Q()
        return float(x.T @ Q @ x)

    def local_solver(self, t0, y: np.ndarray | None = None, *args, **kwargs):
        """
        Default local solver that uses IPOPT to solve the QCQP problem defined by Q and the constraints matrices.
        Consider overwriting this for more efficient solvers.
        """
        print(
            "Warning: using default local_solver, which may be less efficient than a custom one."
        )
        print("Ignoring args and kwargs:", args, kwargs)
        from cert_tools.sdp_solvers import solve_low_rank_sdp

        if y is not None:
            Q = self.get_Q_from_y(y)
        else:
            Q = self.get_Q()

        B = self.get_B_known()
        if len(B) > 0:
            raise NotImplementedError(
                "Inequality constraints are not currently considered by default solver. Must implement your own."
            )

        Constraints = self.get_A_b_list(A_list=self.get_A_known())
        x0 = self.get_x(theta=t0)
        X, info = solve_low_rank_sdp(
            Q, Constraints=Constraints, rank=1, verbose=True, x_cand=x0
        )
        # TODO(FD) identify when the solve is not successful.
        info["success"] = True
        try:
            theta = self.get_theta(X[:, 0])
        except AttributeError:
            theta = X[1 : 1 + self.d, 0]
        return theta, info, info["cost"]

    @property
    def param_dict_landmarks(self):
        assert self.n_parameters is not None

        param_dict = {self.HOM: 1}
        if self.param_level == "no":
            return param_dict
        if self.param_level == "p":
            param_dict.update({f"p_{i}": self.d for i in range(self.n_parameters)})
        if self.param_level == "ppT":
            # Note that ppT is actually
            # [p; vech(ppT)] (linear and quadratic terms)
            # TODO(FD): rename ppT to quadratic
            param_dict.update(
                {
                    f"p_{i}": self.d + int(self.d * (self.d + 1) / 2)
                    for i in range(self.n_parameters)
                }
            )
        return param_dict

    def get_theta(self, x):
        """Inverse of get_x: given lifted vector x, extract elements corresponding to theta.

        Note that x should not contain the homogeinized element x[0] = 1
        """
        assert np.ndim(x) == 1 or x.shape[1] == 1
        if abs(x[0] - 1) < 1e-10:
            print(
                "Warning: got homogenized vector x. The convention is that get_theta should get x[1:]."
                "Please make sure that you use get_theta as intended."
            )
        return x.flatten()[: self.d]
